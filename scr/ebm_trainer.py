# src/ebm_trainer.py

import torch
from torch.utils.data import Dataset, DataLoader
from ebm_model import margin_based_loss
from advanced_neg_sampling import mine_hard_negatives

class HardNegDataset(Dataset):
    """
    Wraps a list of (pos_req_vec, pos_code_vec, neg_req_vec, neg_code_vec)
    for margin-based training.
    """
    def __init__(self, quadruples):
        self.quadruples = quadruples

    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, idx):
        pos_req_vec, pos_code_vec, neg_req_vec, neg_code_vec = self.quadruples[idx]
        return pos_req_vec, pos_code_vec, neg_req_vec, neg_code_vec

def train_ebm_with_hard_negatives(energy_model, embedder, memory_store, 
                                  epochs=5, batch_size=8, lr=1e-3, device='cpu'):
    """
    - Gathers positives & negatives from memory_store
    - Mines 'hard negatives'
    - Trains in a margin-based manner
    """
    positives = memory_store.get_positive_samples()
    negatives = memory_store.get_negative_samples()
    if len(positives) == 0 or len(negatives) == 0:
        print("[train_ebm_with_hard_negatives] No positives or negatives. Skipping.")
        return
    
    # 1) Gather hard negative pairs
    quadruples = mine_hard_negatives(energy_model, embedder, positives, negatives, device=device)
    if len(quadruples) == 0:
        print("[train_ebm_with_hard_negatives] Could not mine any hard negatives.")
        return

    # 2) Create dataset & dataloader
    dataset = HardNegDataset(quadruples)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3) Training
    optimizer = torch.optim.Adam(energy_model.parameters(), lr=lr)
    energy_model.train().to(device)

    for e in range(epochs):
        total_loss = 0.0
        for batch in loader:
            # batch is a tuple of 4 Tensors: each shape [B, embed_dim]
            pos_req_vec, pos_code_vec, neg_req_vec, neg_code_vec = batch

            pos_req_vec = pos_req_vec.to(device)
            pos_code_vec = pos_code_vec.to(device)
            neg_req_vec = neg_req_vec.to(device)
            neg_code_vec = neg_code_vec.to(device)

            # Evaluate energies
            energy_pos = energy_model(pos_code_vec, pos_req_vec)
            energy_neg = energy_model(neg_code_vec, neg_req_vec)
            loss = margin_based_loss(energy_pos, energy_neg, margin=1.0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"[EBM Trainer] Epoch {e+1}/{epochs}, avg_loss={avg_loss:.4f}")
