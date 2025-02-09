# src/train_custom.py

import torch
from ebm_model import EnergyModel
from embedder import CodeRequirementEmbedder
from dataset_trainer import DatasetTrainer

def main():
    # Initialize components
    device = 'cpu'
    
    # Initialize embedder and energy model
    embedder = CodeRequirementEmbedder(
        code_model_name="microsoft/codebert-base",
        req_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )
    
    # Get embedding dimensions
    code_dim, req_dim = embedder.get_embedding_dims()
    
    # Initialize energy model
    energy_model = EnergyModel(
        embed_dim=max(code_dim, req_dim),
        hidden_dim=256
    ).to(device)
    
    # Create trainer
    trainer = DatasetTrainer(
        embedder=embedder,
        energy_model=energy_model,
        device=device
    )
    
    # Train on your dataset
    trainer.train_on_dataset(
        dataset_path="your_dataset.json",
        epochs=5,
        batch_size=4  # Small batch size for CPU
    )
    
    # Save the trained model
    trainer.save_model("trained_ebm.pt")

if __name__ == "__main__":
    main()