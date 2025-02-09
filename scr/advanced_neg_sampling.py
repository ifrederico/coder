# src/advanced_neg_sampling.py

import torch
from torch.utils.data import Dataset, DataLoader

def mine_hard_negatives(energy_model, embedder, positives, negatives, 
                        batch_size=8, device='cpu'):
    """
    positives: list of (req_str, code_str) that passed
    negatives: list of (req_str, code_str) that failed

    Returns a list of (pos_req_emb, pos_code_emb, neg_req_emb, neg_code_emb)
    for "hard negatives" that the model is currently uncertain about.
    """
    # If either is empty, no mining possible
    if len(positives) == 0 or len(negatives) == 0:
        return []

    # Pre-embed all negative samples
    neg_embs = []
    for req_str, code_str in negatives:
        req_vec = embedder.embed_req(req_str).to(device)
        code_vec = embedder.embed_code(code_str).to(device)
        neg_embs.append((req_vec, code_vec))

    # We'll store (pos_req_emb, pos_code_emb, hard_neg_req_emb, hard_neg_code_emb)
    hard_pairs = []

    for pos_req_str, pos_code_str in positives:
        pos_req_vec = embedder.embed_req(pos_req_str).to(device)
        pos_code_vec = embedder.embed_code(pos_code_str).to(device)

        # We'll search for the negative with minimal E(neg) - E(pos)
        with torch.no_grad():
            pos_energy = energy_model(pos_code_vec.unsqueeze(0), 
                                      pos_req_vec.unsqueeze(0)).item()

        best_diff = float('inf')
        best_neg = None

        for neg_req_vec, neg_code_vec in neg_embs:
            with torch.no_grad():
                neg_energy = energy_model(neg_code_vec.unsqueeze(0), 
                                          neg_req_vec.unsqueeze(0)).item()
            diff = neg_energy - pos_energy
            if diff < best_diff:
                best_diff = diff
                best_neg = (neg_req_vec, neg_code_vec)

        if best_neg:
            hard_pairs.append((pos_req_vec, pos_code_vec, best_neg[0], best_neg[1]))
    
    return hard_pairs
