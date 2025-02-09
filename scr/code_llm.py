import os
import asyncio
import aiohttp
import backoff
from typing import Optional, Dict, Any, List
from tenacity import retry, stop_after_attempt, wait_exponential

class CodeLLM:
    """Enhanced wrapper for LLM code generation with async support and error handling"""
    
    def __init__(
        self,
        model_name: str = "gpt-4-turbo-preview",
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        max_retries: int = 3,
        timeout: int = 30,
    ):
        """
        Initialize the CodeLLM wrapper.
        
        Args:
            model_name: Name of the model to use
            api_key: OpenAI API key (defaults to env var)
            api_base: Base URL for API
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout in seconds for API calls
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("No API key provided and OPENAI_API_KEY env var not set")
        
        self.api_base = api_base
        self.max_retries = max_retries
        self.timeout = timeout
        self._session = None
        
        # System prompt for code generation
        self.system_prompt = """You are an expert Python programmer. 
        Generate clean, efficient, and well-structured code that meets the given requirements.
        Return only the code without any explanations or comments unless specifically requested."""
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=3
    )
    async def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> Dict[str, Any]:
        """Make API request with retries and error handling"""
        session = await self._get_session()
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        async with session.post(f"{self.api_base}/chat/completions", json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                raise ValueError(f"API request failed: {error_text}")
            return await response.json()

    async def generate_code_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """
        Asynchronously generate code using the LLM.
        
        Args:
            prompt: The coding task description
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens in the response
            **kwargs: Additional parameters for the API call
            
        Returns:
            str: Generated code
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = await self._make_request(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Error generating code: {str(e)}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_code(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """
        Synchronous version of generate_code_async for compatibility.
        Uses asyncio.run() internally.
        """
        return asyncio.run(
            self.generate_code_async(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        )

    async def __aenter__(self):
        """Async context manager entry"""
        await self._get_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session and not self._session.closed:
            await self._session.close()

# Example usage
if __name__ == "__main__":
    async def main():
        llm = CodeLLM()
        prompt = "Write a Python function that implements binary search"
        
        async with llm:
            code = await llm.generate_code_async(prompt)
            print(code)
    
    asyncio.run(main())