# src/self_coding_agent.py

import torch
from test_runner import run_tests_on_code
from ebm_trainer import train_ebm_with_hard_negatives

class SelfCodingAgent:
    def __init__(self, llm, embedder, energy_model, memory_store, device='cpu'):
        self.llm = llm
        self.embedder = embedder
        self.energy_model = energy_model
        self.memory = memory_store
        self.device = device

    def solve_task(self, requirement_str, iterations=3, train_after=True):
        """
        For demonstration: 
          - Generate code from LLM
          - Test it
          - Store in memory
          - (Optionally) retrain EBM afterwards
        """
        for i in range(iterations):
            # Simple prompt
            prompt = f"Requirement:\n{requirement_str}\nWrite Python code."
            code_candidate = self.llm.generate_code(prompt, max_tokens=300)

            passed, logs = run_tests_on_code(code_candidate)
            self.memory.add_record(requirement_str, code_candidate, passed)

            if passed:
                print(f"[SelfCodingAgent] Code passed on attempt {i+1}")
                # Optionally break early if you only need one success
                break
            else:
                print(f"[SelfCodingAgent] Code failed on attempt {i+1}, logs:\n{logs}")

        # Train EBM after collecting some new data
        if train_after:
            train_ebm_with_hard_negatives(self.energy_model, self.embedder, self.memory,
                                          epochs=5, device=self.device)

        # Return last candidate (passed or not)
        return code_candidate

