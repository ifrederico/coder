# src/main.py

import torch
from code_llm import CodeLLM
from ebm_model import EnergyModel
from embedder import Embedder
from memory_store import MemoryStore
from self_coding_agent import SelfCodingAgent

def main():
    device = 'cpu'
    print("[main] Starting advanced EBM-based self-coding demo.")

    # 1) Instantiate components
    llm = CodeLLM(model_name="gpt-3.5-turbo")
    embedder = Embedder(embed_dim=64)  # smaller for demonstration
    energy_model = EnergyModel(embed_dim=64, hidden_dim=128).to(device)
    memory_store = MemoryStore()

    # 2) Create self-coding agent
    agent = SelfCodingAgent(llm, embedder, energy_model, memory_store, device=device)

    # 3) Example coding tasks
    tasks = [
        "Write a Python function 'greet' that prints 'Hello, negative sampling!'.",
        "Create a function 'add(x, y)' that returns x + y. Must pass syntax check.",
    ]

    for i, task in enumerate(tasks, 1):
        print(f"\n=== Task {i}: {task} ===")
        final_code = agent.solve_task(task, iterations=3, train_after=True)
        print("[main] Final code:\n", final_code)

    print("[main] Done.")

if __name__ == "__main__":
    main()
