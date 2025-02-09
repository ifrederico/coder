import torch
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from test_runner import run_tests_on_code
from ebm_trainer import train_ebm_with_hard_negatives

@dataclass
class CodeCandidate:
    """Represents a single code candidate with its associated metadata"""
    code: str
    energy: float
    requirement: str
    passed_tests: bool = False
    test_logs: str = ""
    
class EnhancedSelfCodingAgent:
    def __init__(
        self,
        llm,
        embedder,
        energy_model,
        memory_store,
        device='cpu',
        num_candidates: int = 5,
        max_workers: int = 3,
        temperature_range: Tuple[float, float] = (0.3, 0.9)
    ):
        self.llm = llm
        self.embedder = embedder
        self.energy_model = energy_model
        self.memory = memory_store
        self.device = device
        self.num_candidates = num_candidates
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.temp_range = temperature_range

    async def generate_candidates(
        self,
        requirement: str,
        prompt_template: Optional[str] = None
    ) -> List[CodeCandidate]:
        """Generate multiple code candidates with varying temperatures"""
        if prompt_template is None:
            prompt_template = (
                "Write Python code that meets this requirement:\n"
                "{requirement}\n\n"
                "Return only the code without any explanations or comments."
            )

        # Generate code with different temperatures for diversity
        temperatures = torch.linspace(*self.temp_range, self.num_candidates).tolist()
        
        async def generate_single(temp: float) -> str:
            prompt = prompt_template.format(requirement=requirement)
            return await self.llm.generate_code_async(
                prompt,
                max_tokens=500,
                temperature=temp
            )

        # Generate candidates in parallel
        candidates = await asyncio.gather(
            *[generate_single(temp) for temp in temperatures]
        )
        
        # Compute energies for all candidates
        code_candidates = []
        for code in candidates:
            energy = await self.compute_energy(code, requirement)
            code_candidates.append(CodeCandidate(
                code=code,
                energy=energy,
                requirement=requirement
            ))
            
        # Sort by energy (lower is better)
        return sorted(code_candidates, key=lambda x: x.energy)

    async def compute_energy(self, code: str, requirement: str) -> float:
        """Compute EBM energy score for a code-requirement pair"""
        with torch.no_grad():
            code_emb = self.embedder.embed_code(code).to(self.device)
            req_emb = self.embedder.embed_requirement(requirement).to(self.device)
            
            energy = self.energy_model(
                code_emb.unsqueeze(0),
                req_emb.unsqueeze(0)
            ).item()
            
        return energy

    async def test_candidates(
        self,
        candidates: List[CodeCandidate]
    ) -> List[CodeCandidate]:
        """Run tests on candidates in parallel"""
        def test_single(candidate: CodeCandidate) -> CodeCandidate:
            passed, logs = run_tests_on_code(candidate.code)
            candidate.passed_tests = passed
            candidate.test_logs = logs
            return candidate

        # Run tests in thread pool
        loop = asyncio.get_event_loop()
        tested_candidates = await asyncio.gather(
            *[loop.run_in_executor(self.executor, test_single, c)
              for c in candidates]
        )
        
        return tested_candidates

    def update_memory(self, candidates: List[CodeCandidate]):
        """Store results in memory for EBM training"""
        for candidate in candidates:
            self.memory.add_record(
                candidate.requirement,
                candidate.code,
                candidate.passed_tests
            )

    async def solve_task(
        self,
        requirement: str,
        max_iterations: int = 3,
        train_after: bool = True,
        prompt_template: Optional[str] = None
    ) -> Dict:
        """Main solving loop with multi-candidate generation and testing"""
        best_candidate = None
        all_candidates = []
        
        for iteration in range(max_iterations):
            print(f"\n[Iteration {iteration + 1}/{max_iterations}]")
            
            # Generate candidates
            candidates = await self.generate_candidates(
                requirement,
                prompt_template
            )
            
            # Test all candidates
            tested_candidates = await self.test_candidates(candidates)
            all_candidates.extend(tested_candidates)
            
            # Update best candidate if we found a passing solution
            passing_candidates = [c for c in tested_candidates if c.passed_tests]
            if passing_candidates:
                best_candidate = min(
                    passing_candidates,
                    key=lambda x: x.energy
                )
                print(f"Found passing solution with energy: {best_candidate.energy:.4f}")
                break
                
            print(f"No passing solutions this iteration. Best energy: {candidates[0].energy:.4f}")
        
        # Store results and train
        self.update_memory(all_candidates)
        if train_after:
            train_ebm_with_hard_negatives(
                self.energy_model,
                self.embedder,
                self.memory,
                epochs=5,
                device=self.device
            )
            
        return {
            "best_candidate": best_candidate,
            "all_candidates": all_candidates,
            "num_iterations": iteration + 1,
            "found_solution": best_candidate is not None
        }

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.executor.shutdown()