# src/ebm_model.py

import torch
import torch.nn as nn
import torch.optim as optim

class EnergyModel(nn.Module):
    """
    E(code_emb, req_emb) => scalar (energy).
    Lower => code & requirement are more compatible.
    """
    def __init__(self, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, code_emb, req_emb):
        combined = torch.cat([code_emb, req_emb], dim=-1)
        energy = self.net(combined).squeeze(-1)
        return energy

def margin_based_loss(energy_pos, energy_neg, margin=1.0):
    """
    L = mean( max(0, margin + E(pos) - E(neg)) )
    We want E(pos) + margin <= E(neg).
    """
    loss = torch.relu(margin + energy_pos - energy_neg)
    return torch.mean(loss)
