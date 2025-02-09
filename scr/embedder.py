# src/embedder.py

import torch
from transformers import AutoTokenizer, AutoModel
from typing import Union, Tuple
import numpy as np

class CodeRequirementEmbedder:
    def __init__(
        self,
        code_model_name: str = "microsoft/codebert-base",
        req_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """Initialize separate embedders for code and requirements."""
        self.device = device
        
        # Initialize code embedder
        self.code_tokenizer = AutoTokenizer.from_pretrained(code_model_name)
        self.code_model = AutoModel.from_pretrained(
            code_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
        # Initialize requirement embedder
        self.req_tokenizer = AutoTokenizer.from_pretrained(req_model_name)
        self.req_model = AutoModel.from_pretrained(
            req_model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        ).to(device)
        
        # Set embedding dimension based on model outputs
        self.code_embed_dim = self.code_model.config.hidden_size
        self.req_embed_dim = self.req_model.config.hidden_size
        
    def embed_code(self, code: str) -> torch.Tensor:
        """Generate embeddings for code snippets."""
        inputs = self.code_tokenizer(
            code,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.code_model(**inputs)
            code_embedding = outputs.last_hidden_state[:, 0, :]
            
        return code_embedding
    
    def embed_req(self, requirement: str) -> torch.Tensor:
        """Generate embeddings for requirement specifications."""
        inputs = self.req_tokenizer(
            requirement,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.req_model(**inputs)
            mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            req_embedding = torch.sum(token_embeddings * mask, dim=1) / torch.sum(mask, dim=1)
            
        return req_embedding
        
    def get_embedding_dims(self) -> Tuple[int, int]:
        """Get the dimensions of code and requirement embeddings."""
        return self.code_embed_dim, self.req_embed_dim

    @staticmethod
    def normalize_embeddings(
        embeddings: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """L2 normalize embeddings."""
        if isinstance(embeddings, tuple):
            return (
                torch.nn.functional.normalize(embeddings[0], p=2, dim=1),
                torch.nn.functional.normalize(embeddings[1], p=2, dim=1)
            )
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)