# src/code_llm.py

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional
import asyncio
import gc

class CodeLLM:
    """Local LLM wrapper using HuggingFace models"""
    
    def __init__(
        self,
        model_name: str = "microsoft/phi-1_5",  # More memory efficient model
        device: str = "cpu",  # Use CPU for stability
        max_length: int = 2048
    ):
        """Initialize local LLM."""
        self.device = device
        self.max_length = max_length
        
        print(f"Loading model {model_name} on {device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Load model with memory optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            offload_folder="model_cache"  # Cache large tensors to disk
        ).to(device)
        
        # Make sure we have a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # System prompt for code generation
        self.system_prompt = """Write Python code for the following task.
        Return only the code without any explanations.
        The code should be clean, efficient, and well-structured.
        
        Task description:
        """
    
    async def generate_code_async(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Asynchronously generate code using the local model."""
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Combine system prompt and user prompt
            full_prompt = f"{self.system_prompt}\n{prompt}\n\nCode:\n"
            
            # Tokenize input
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length // 2,
                padding=True
            ).to(self.device)
            
            # Set up generation parameters
            gen_kwargs = {
                "max_length": max_tokens or self.max_length,
                "temperature": temperature,
                "do_sample": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "num_return_sequences": 1,
                "top_p": 0.95,
                "top_k": 50,
                **kwargs
            }
            
            # Run generation in executor to avoid blocking
            loop = asyncio.get_event_loop()
            tokens = await loop.run_in_executor(
                None,
                lambda: self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **gen_kwargs
                )
            )
            
            # Decode and clean up response
            generated_code = self.tokenizer.decode(tokens[0], skip_special_tokens=True)
            
            # Extract only the code part after our prompt
            code_part = generated_code[len(full_prompt):].strip()
            
            # Clean up any remaining prompt artifacts
            if code_part.startswith("python") or code_part.startswith("Python"):
                code_part = code_part[code_part.find("\n")+1:]
            
            # Force garbage collection again
            del tokens, inputs
            gc.collect()
            
            return code_part.strip()
            
        except Exception as e:
            print(f"Error during code generation: {str(e)}")
            return f"# Error during generation: {str(e)}"
    
    def generate_code(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """Synchronous version of generate_code_async"""
        return asyncio.run(
            self.generate_code_async(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clean up resources
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()