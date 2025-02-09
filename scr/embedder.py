# src/embedder.py

import torch

class Embedder:
    """
    Converts code_str or requirement_str into a vector of size embed_dim.
    Replace the dummy approach with a real embedding model if possible.
    """
    def __init__(self, embed_dim=128):
        self.embed_dim = embed_dim

    def embed_code(self, code_str):
        # Dummy: create random vector (deterministic for demonstration?).
        # Real approach might be: "transformer.encode(code_str)".
        return torch.randn(self.embed_dim)

    def embed_req(self, req_str):
        return torch.randn(self.embed_dim)
