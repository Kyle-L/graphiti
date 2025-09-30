"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import os
import typing
from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Dict, List, Literal

from pydantic import BaseModel, ValidationError

from ..prompts.models import Message
from .client import LLMClient
from .config import DEFAULT_MAX_TOKENS, LLMConfig, ModelSize
from .errors import RateLimitError, RefusalError

if TYPE_CHECKING:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
else:
    try:
        import boto3
        from botocore.config import Config
        from botocore.exceptions import ClientError
    except ImportError:
        raise ImportError(
            'boto3 is required for BedrockClient. '
            'Install it with: pip install boto3'
        ) from None


logger = logging.getLogger(__name__)

BedrockModel = Literal[
    'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    'us.anthropic.claude-3-5-sonnet-20241022-v2:0',
    'us.anthropic.claude-3-5-sonnet-20240620-v1:0',
    'us.anthropic.claude-3-5-haiku-20241022-v1:0',
    'us.anthropic.claude-3-opus-20240229-v1:0',
    'us.anthropic.claude-3-sonnet-20240229-v1:0',
    'us.anthropic.claude-3-haiku-20240307-v1:0',
    'anthropic.claude-3-7-sonnet-20250219-v1:0',
    'anthropic.claude-3-5-sonnet-20241022-v2:0',
    'anthropic.claude-3-5-sonnet-20240620-v1:0',
    'anthropic.claude-3-5-haiku-20241022-v1:0',
    'anthropic.claude-3-opus-20240229-v1:0',
    'anthropic.claude-3-sonnet-20240229-v1:0',
    'anthropic.claude-3-haiku-20240307-v1:0',
    'amazon.nova-pro-v1:0',
    'amazon.nova-lite-v1:0',
    'amazon.nova-micro-v1:0',
]

DEFAULT_MODEL: BedrockModel = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'


class BedrockClient(LLMClient):
    """
    A client for the Amazon Bedrock LLM service.

    Args:
        config: A configuration object for the LLM.
        cache: Whether to cache the LLM responses.
        client: An optional boto3 bedrock-runtime client instance to use.
        max_tokens: The maximum number of tokens to generate.

    Methods:
        generate_response: Generate a response from the LLM.

    Notes:
        - If a LLMConfig is not provided, AWS credentials will be pulled from the standard AWS credential chain,
          and all default values will be used for the LLMConfig.
        - The client uses the boto3 bedrock-runtime service with extended timeouts for long-running requests.
    """

    model: BedrockModel

    def __init__(
        self,
        config: LLMConfig | None = None,
        cache: bool = False,
        client: Any | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        if config is None:
            config = LLMConfig()
            config.max_tokens = max_tokens

        if config.model is None:
            config.model = DEFAULT_MODEL

        super().__init__(config, cache)
        # Explicitly set the instance model to the config model to prevent type checking errors
        self.model = typing.cast(BedrockModel, config.model)

        if not client:
            # Create a boto3 config with extended timeout for Bedrock
            bedrock_config = Config(
                read_timeout=14 * 60,  # 14 minutes
                retries={"max_attempts": 3}
            )
            self.client = boto3.client(
                service_name="bedrock-runtime",
                config=bedrock_config
            )
        else:
            self.client = client

    def _extract_json_from_text(self, text: str) -> dict[str, typing.Any]:
        """Extract JSON from text content.

        A helper method to extract JSON from text content, used when tool use fails or
        no response_model is provided.

        Args:
            text: The text to extract JSON from

        Returns:
            Extracted JSON as a dictionary

        Raises:
            ValueError: If JSON cannot be extracted or parsed
        """
        try:
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = text[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError(f'Could not extract JSON from model response: {text}')
        except (JSONDecodeError, ValueError) as e:
            raise ValueError(f'Could not extract JSON from model response: {text}') from e

    def _create_tool(
        self, response_model: type[BaseModel] | None = None
    ) -> List[Dict[str, Any]]:
        """
        Create a tool definition based on the response_model if provided, or a generic JSON tool if not.

        Args:
            response_model: Optional Pydantic model to use for structured output.

        Returns:
            A list containing tool definitions for use with the Bedrock API.
        """
        if response_model is not None:
            # Use the response_model to define the tool
            model_schema = response_model.model_json_schema()
            tool_name = response_model.__name__
            description = model_schema.get('description', f'Extract {tool_name} information')
        else:
            # Create a generic JSON output tool
            tool_name = 'generic_json_output'
            description = 'Output data in JSON format'
            model_schema = {
                'type': 'object',
                'additionalProperties': True,
                'description': 'Any JSON object containing the requested information',
            }

        bedrock_tool = {
            "toolSpec": {
                "name": tool_name,
                "description": description,
                "inputSchema": {"json": model_schema},
            }
        }
        return [bedrock_tool]

    async def _generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the Bedrock LLM using tool-based approach for all requests.

        Args:
            messages: List of message objects to send to the LLM.
            response_model: Optional Pydantic model to use for structured output.
            max_tokens: Maximum number of tokens to generate.
            model_size: Model size preference (unused in Bedrock, kept for compatibility).

        Returns:
            Dictionary containing the structured response from the LLM.

        Raises:
            RateLimitError: If the rate limit is exceeded.
            RefusalError: If the LLM refuses to respond.
            Exception: If an error occurs during the generation process.
        """
        system_message = messages[0]
        user_messages = messages[1:]

        # Convert messages to Bedrock format
        bedrock_messages = []
        for msg in user_messages:
            bedrock_messages.append({
                "role": msg.role,
                "content": [{"text": msg.content}]
            })

        # Set up inference configuration
        max_creation_tokens: int = min(
            max_tokens if max_tokens is not None else self.config.max_tokens,
            DEFAULT_MAX_TOKENS,
        )

        inference_config = {
            "maxTokens": max_creation_tokens,
            "temperature": self.temperature,
        }

        try:
            # Create the appropriate tool based on whether response_model is provided
            tools = self._create_tool(response_model)
            
            bedrock_request = {
                "modelId": self.model,
                "messages": bedrock_messages,
                "inferenceConfig": inference_config,
                "system": [{"text": system_message.content}],
                "toolConfig": {"tools": tools}
            }

            response = self.client.converse(**bedrock_request)

            # Extract the tool output from the response
            for content_item in response["output"]["message"]["content"]:
                if "toolUse" in content_item:
                    tool_use = content_item["toolUse"]
                    if isinstance(tool_use["input"], dict):
                        tool_args: dict[str, typing.Any] = tool_use["input"]
                    else:
                        tool_args = json.loads(str(tool_use["input"]))
                    return tool_args

            # If we didn't get a proper tool_use response, try to extract from text
            for content_item in response["output"]["message"]["content"]:
                if "text" in content_item:
                    return self._extract_json_from_text(content_item["text"])

            # If we get here, we couldn't parse a structured response
            raise ValueError(
                f'Could not extract structured data from model response: {response["output"]["message"]["content"]}'
            )

        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            
            if error_code == 'ThrottlingException':
                raise RateLimitError(f'Rate limit exceeded. Please try again later. Error: {error_message}') from e
            elif 'ValidationException' in error_code:
                # Check if this is a content policy violation
                if 'content policy' in error_message.lower() or 'refused' in error_message.lower():
                    raise RefusalError(error_message) from e
            raise e
        except Exception as e:
            raise e

    async def generate_response(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        max_tokens: int | None = None,
        model_size: ModelSize = ModelSize.medium,
    ) -> dict[str, typing.Any]:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message objects to send to the LLM.
            response_model: Optional Pydantic model to use for structured output.
            max_tokens: Maximum number of tokens to generate.
            model_size: Model size preference (unused in Bedrock, kept for compatibility).

        Returns:
            Dictionary containing the structured response from the LLM.

        Raises:
            RateLimitError: If the rate limit is exceeded.
            RefusalError: If the LLM refuses to respond.
            Exception: If an error occurs during the generation process.
        """
        if max_tokens is None:
            max_tokens = self.max_tokens

        retry_count = 0
        max_retries = 2
        last_error: Exception | None = None

        while retry_count <= max_retries:
            try:
                response = await self._generate_response(
                    messages, response_model, max_tokens, model_size
                )

                # If we have a response_model, attempt to validate the response
                if response_model is not None:
                    # Validate the response against the response_model
                    model_instance = response_model(**response)
                    return model_instance.model_dump()

                # If no validation needed, return the response
                return response

            except (RateLimitError, RefusalError):
                # These errors should not trigger retries
                raise
            except Exception as e:
                last_error = e

                if retry_count >= max_retries:
                    if isinstance(e, ValidationError):
                        logger.error(
                            f'Validation error after {retry_count}/{max_retries} attempts: {e}'
                        )
                    else:
                        logger.error(f'Max retries ({max_retries}) exceeded. Last error: {e}')
                    raise e

                if isinstance(e, ValidationError):
                    response_model_cast = typing.cast(type[BaseModel], response_model)
                    error_context = f'The previous response was invalid. Please provide a valid {response_model_cast.__name__} object. Error: {e}'
                else:
                    error_context = (
                        f'The previous response attempt was invalid. '
                        f'Error type: {e.__class__.__name__}. '
                        f'Error details: {str(e)}. '
                        f'Please try again with a valid response.'
                    )

                # Common retry logic
                retry_count += 1
                messages.append(Message(role='user', content=error_context))
                logger.warning(f'Retrying after error (attempt {retry_count}/{max_retries}): {e}')

        # If we somehow get here, raise the last error
        raise last_error or Exception('Max retries exceeded with no specific error')