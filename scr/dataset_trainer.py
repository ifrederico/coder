# src/dataset_trainer.py

import json
import torch
from typing import List, Dict, Tuple
from pathlib import Path
from tqdm import tqdm
from ebm_model import EnergyModel, margin_based_loss
from embedder import CodeRequirementEmbedder

class DatasetTrainer:
    """Handles loading and training with custom code datasets"""
    
    def __init__(
        self,
        embedder: CodeRequirementEmbedder,
        energy_model: EnergyModel,
        device: str = 'cpu'
    ):
        self.embedder = embedder
        self.energy_model = energy_model
        self.device = device

    def load_dataset(self, file_path: str) -> List[Dict]:
        """
        Load dataset from a JSON file.
        Expected format:
        [
            {
                "requirement": "Write a function to...",
                "solution": "def my_func()...",
                "is_correct": true
            },
            ...
        ]
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        with open(path, 'r') as f:
            dataset = json.load(f)
            
        print(f"Loaded {len(dataset)} examples from {file_path}")
        return dataset

    def prepare_batch(
        self,
        examples: List[Dict],
        batch_size: int = 8
    ) -> List[Tuple[torch.Tensor, torch.Tensor, bool]]:
        """Prepare a batch of examples for training"""
        batch = []
        
        for i in range(0, len(examples), batch_size):
            batch_examples = examples[i:i + batch_size]
            
            for example in batch_examples:
                # Get embeddings
                req_emb = self.embedder.embed_req(example["requirement"])
                code_emb = self.embedder.embed_code(example["solution"])
                
                batch.append((
                    req_emb.to(self.device),
                    code_emb.to(self.device),
                    example["is_correct"]
                ))
                
        return batch

    def train_on_dataset(
        self,
        dataset_path: str,
        epochs: int = 5,
        batch_size: int = 8,
        learning_rate: float = 1e-4
    ):
        """Train the EBM on a custom dataset"""
        # Load dataset
        dataset = self.load_dataset(dataset_path)
        
        # Prepare optimizer
        optimizer = torch.optim.Adam(
            self.energy_model.parameters(),
            lr=learning_rate
        )
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0
            
            # Prepare batches
            batches = self.prepare_batch(dataset, batch_size)
            
            # Training progress bar
            pbar = tqdm(batches, desc=f"Epoch {epoch+1}/{epochs}")
            
            for req_emb, code_emb, is_correct in pbar:
                # Compute energy
                energy = self.energy_model(code_emb, req_emb)
                
                # Compute loss based on whether example is correct
                if is_correct:
                    # Correct examples should have lower energy
                    loss = torch.mean(energy)
                else:
                    # Incorrect examples should have higher energy
                    loss = -torch.mean(energy)
                    
                # Update weights
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': total_loss / batch_count
                })
            
            avg_loss = total_loss / batch_count
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
    def save_model(self, save_path: str):
        """Save the trained EBM model"""
        torch.save(self.energy_model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
    def load_model(self, load_path: str):
        """Load a trained EBM model"""
        self.energy_model.load_state_dict(torch.load(load_path))
        print(f"Model loaded from {load_path}")

# Example dataset format
EXAMPLE_DATASET = """
[
    {
        "requirement": "Write a function to check if a number is prime",
        "solution": "def is_prime(n: int) -> bool:\\n    if n < 2: return False\\n    for i in range(2, int(n ** 0.5) + 1):\\n        if n % i == 0: return False\\n    return True",
        "is_correct": true
    },
    {
        "requirement": "Write a function to check if a number is prime",
        "solution": "def is_prime(n):\\n    return n > 1",
        "is_correct": false
    }
]
"""