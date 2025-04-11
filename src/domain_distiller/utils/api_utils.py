"""
API utilities for interacting with OpenAI-compatible APIs.
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Union, Callable
import asyncio
import aiohttp
from aiohttp import ClientSession
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logger
logger = logging.getLogger(__name__)


class APIError(Exception):
    """Exception raised for API errors."""
    pass


class RateLimitError(APIError):
    """Exception raised for rate limit errors."""
    pass


class APIClient:
    """Client for interacting with OpenAI-compatible APIs."""
    
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        timeout: int = 60,
        max_retries: int = 5
    ):
        """
        Initialize the API client.
        
        Args:
            api_base: Base URL for the API
            api_key: API key for authentication
            model: Model name to use
            timeout: Timeout for API requests in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_base = api_base.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._session: Optional[ClientSession] = None
        
        logger.debug(f"Initialized API client for {self.api_base} using model {self.model}")
    
    async def ensure_session(self) -> ClientSession:
        """Ensure that the API client has a session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async def close(self) -> None:
        """Close the API client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    def _get_endpoint(self, endpoint_type: str) -> str:
        """Get the API endpoint for the specified type."""
        # Default OpenAI-compatible endpoints
        endpoints = {
            "chat": "/chat/completions",
            "completions": "/completions",
            "embeddings": "/embeddings"
        }
        
        # Handle different API formats
        if "openai" in self.api_base:
            # Standard OpenAI API format
            return f"{self.api_base}{endpoints[endpoint_type]}"
        elif "anthropic" in self.api_base:
            # Anthropic API (Claude) format
            if endpoint_type == "chat":
                return f"{self.api_base}/v1/messages"
            else:
                raise ValueError(f"Anthropic API does not support endpoint type: {endpoint_type}")
        else:
            # Generic OpenAI-compatible API format (assume standard paths)
            return f"{self.api_base}{endpoints[endpoint_type]}"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get the headers for API requests."""
        if "anthropic" in self.api_base:
            return {
                "x-api-key": self.api_key,
                "content-type": "application/json",
                "anthropic-version": "2023-06-01"  # Use appropriate version
            }
        else:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
    
    def _format_payload(self, endpoint_type: str, **kwargs) -> Dict[str, Any]:
        """Format the payload for API requests based on the endpoint type and API provider."""
        if "anthropic" in self.api_base:
            # Format for Anthropic API
            if endpoint_type == "chat":
                messages = kwargs.get("messages", [])
                system_message = None
                user_messages = []
                
                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    elif msg["role"] == "user":
                        user_messages.append(msg["content"])
                
                # Combine all user messages if multiple (not ideal but a fallback)
                user_content = "\n".join(user_messages) if user_messages else ""
                
                return {
                    "model": self.model,
                    "messages": [
                        {"role": "user", "content": user_content}
                    ],
                    "system": system_message,
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "temperature": kwargs.get("temperature", 0.7)
                }
        
        # Default format for OpenAI API
        return {
            "model": self.model,
            **{k: v for k, v in kwargs.items() if v is not None}
        }
    
    def _parse_response(self, endpoint_type: str, response_data: Dict[str, Any]) -> Any:
        """Parse the API response based on the endpoint type and API provider."""
        if "anthropic" in self.api_base:
            # Parse Anthropic API response
            if endpoint_type == "chat":
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": response_data.get("content", [{"text": ""}])[0]["text"]
                            }
                        }
                    ]
                }
        
        # Return OpenAI-formatted response as is
        return response_data
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(RateLimitError)
    )
    async def _make_request(
        self,
        endpoint_type: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Make a request to the API with retry logic."""
        session = await self.ensure_session()
        endpoint = self._get_endpoint(endpoint_type)
        headers = self._get_headers()
        
        try:
            async with session.post(endpoint, json=payload, headers=headers) as response:
                response_text = await response.text()
                
                # Handle rate limiting
                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", "1"))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds.")
                    raise RateLimitError(f"Rate limited: {response_text}")
                
                # Handle other errors
                if response.status != 200:
                    logger.error(f"API error: {response.status} - {response_text}")
                    raise APIError(f"API returned error {response.status}: {response_text}")
                
                response_data = json.loads(response_text)
                return self._parse_response(endpoint_type, response_data)
                
        except aiohttp.ClientError as e:
            logger.error(f"Client error: {str(e)}")
            raise APIError(f"Client error: {str(e)}")
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        response_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make a chat completion request.
        
        Args:
            messages: List of message objects with role and content
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            response_format: Format specification for the response
            
        Returns:
            API response with completion
        """
        payload = self._format_payload(
            "chat",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            response_format=response_format
        )
        
        return await self._make_request("chat", payload)
    
    async def get_json_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2000,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a JSON response from the API.
        
        Args:
            messages: List of message objects with role and content
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            schema: JSON schema to validate the response against
            
        Returns:
            Parsed JSON response
        """
        # Configure response format for JSON
        response_format = {"type": "json_object"}
        if schema:
            # Some APIs support schema validation
            response_format["schema"] = schema
        
        response = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format
        )
        
        try:
            if "choices" in response and len(response["choices"]) > 0:
                if "message" in response["choices"][0]:
                    content = response["choices"][0]["message"]["content"]
                    return json.loads(content)
            
            # Fallback if standard format isn't found
            if "content" in response and isinstance(response["content"], list):
                for content_item in response["content"]:
                    if content_item.get("type") == "text":
                        return json.loads(content_item["text"])
            
            raise APIError("Unexpected response format")
        
        except json.JSONDecodeError:
            logger.error("Failed to parse JSON response")
            raise APIError("Failed to parse JSON response")
    
    async def structured_generation(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        schema: Dict[str, Any] = None,
        temperature: float = 0.2,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Generate structured content based on a schema.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            schema: JSON schema defining the response structure
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Structured response conforming to the schema
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        return await self.get_json_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            schema=schema
        )


class AsyncWorkerPool:
    """Pool of async workers for parallel API requests."""
    
    def __init__(
        self,
        api_client: APIClient,
        num_workers: int = 4,
        rate_limit: int = 10,  # requests per minute
        batch_size: int = 5
    ):
        """
        Initialize the worker pool.
        
        Args:
            api_client: API client to use for requests
            num_workers: Number of parallel workers
            rate_limit: Maximum requests per minute
            batch_size: Number of items to process in each batch
        """
        self.api_client = api_client
        self.num_workers = num_workers
        self.rate_limit = rate_limit
        self.batch_size = batch_size
        self.semaphore = asyncio.Semaphore(num_workers)
        self.request_times = []
        
        logger.debug(f"Initialized worker pool with {num_workers} workers")
    
    async def _enforce_rate_limit(self) -> None:
        """Enforce the rate limit by waiting if necessary."""
        now = time.time()
        
        # Remove request times older than 60 seconds
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        # If at rate limit, wait until we can make another request
        if len(self.request_times) >= self.rate_limit:
            oldest = min(self.request_times)
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
        
        # Add current time to request times
        self.request_times.append(time.time())
    
    async def _worker(
        self,
        item: Any,
        process_func: Callable[[APIClient, Any], Any]
    ) -> Any:
        """
        Worker function to process an item with rate limiting.
        
        Args:
            item: Item to process
            process_func: Function to process the item
            
        Returns:
            Processed result
        """
        async with self.semaphore:
            try:
                await self._enforce_rate_limit()
                result = await process_func(self.api_client, item)
                return result
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                return {"error": str(e), "item": item}
    
    async def process_items(
        self,
        items: List[Any],
        process_func: Callable[[APIClient, Any], Any]
    ) -> List[Any]:
        """
        Process a list of items in parallel.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            
        Returns:
            List of processed results
        """
        tasks = []
        for item in items:
            task = asyncio.ensure_future(self._worker(item, process_func))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def process_in_batches(
        self,
        items: List[Any],
        process_func: Callable[[APIClient, Any], Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Process a list of items in batches.
        
        Args:
            items: List of items to process
            process_func: Function to process each item
            batch_size: Size of each batch (defaults to self.batch_size)
            
        Returns:
            List of processed results
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        all_results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(items) + batch_size - 1) // batch_size}")
            batch_results = await self.process_items(batch, process_func)
            all_results.extend(batch_results)
        
        return all_results
    
    async def close(self) -> None:
        """Close the worker pool and the API client."""
        await self.api_client.close()
