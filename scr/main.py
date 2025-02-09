import asyncio
import torch
from code_llm import CodeLLM
from ebm_model import EnergyModel
from embedder import CodeRequirementEmbedder  # Using our enhanced embedder
from memory_store import MemoryStore
from enhanced_self_coding_agent import EnhancedSelfCodingAgent
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Initialize components with enhanced versions
    llm = CodeLLM(model_name="gpt-4-turbo-preview")
    
    # Initialize our enhanced embedder
    embedder = CodeRequirementEmbedder(
        code_model_name="microsoft/codebert-base",
        req_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )
    
    # Get embedding dimensions from the embedder
    code_dim, req_dim = embedder.get_embedding_dims()
    
    # Initialize energy model with correct dimensions
    energy_model = EnergyModel(
        embed_dim=max(code_dim, req_dim),
        hidden_dim=256
    ).to(device)
    
    memory_store = MemoryStore()

    # Create enhanced self-coding agent
    async with EnhancedSelfCodingAgent(
        llm=llm,
        embedder=embedder,
        energy_model=energy_model,
        memory_store=memory_store,
        device=device,
        num_candidates=5  # Generate 5 candidates per iteration
    ) as agent:
        
        # Example coding tasks with increasing complexity
        tasks = [
            {
                "requirement": "Write a Python function that implements bubble sort",
                "template": """
                Write a Python function that implements bubble sort.
                The function should:
                1. Take a list of numbers as input
                2. Return the sorted list
                3. Use the bubble sort algorithm
                4. Include proper type hints
                Return only the code without explanations.
                """
            },
            {
                "requirement": "Create a cache decorator with timeout",
                "template": """
                Create a Python decorator that implements a caching mechanism with timeout.
                Requirements:
                1. Cache function results
                2. Support timeout (expire cached results after N seconds)
                3. Use proper type hints
                4. Thread-safe implementation
                Return only the code without explanations.
                """
            }
        ]

        for task in tasks:
            logger.info(f"\n=== Starting task: {task['requirement']} ===")
            
            result = await agent.solve_task(
                requirement=task['requirement'],
                prompt_template=task['template'],
                max_iterations=3
            )
            
            if result['found_solution']:
                best_code = result['best_candidate'].code
                logger.info(f"Found solution in {result['num_iterations']} iterations")
                logger.info(f"Best solution energy: {result['best_candidate'].energy:.4f}")
                logger.info("Code:\n" + best_code)
            else:
                logger.warning("No passing solution found")
                
            # Log statistics about all candidates
            all_candidates = result['all_candidates']
            pass_rate = sum(1 for c in all_candidates if c.passed_tests) / len(all_candidates)
            avg_energy = sum(c.energy for c in all_candidates) / len(all_candidates)
            
            logger.info(f"Statistics:")
            logger.info(f"- Pass rate: {pass_rate:.2%}")
            logger.info(f"- Average energy: {avg_energy:.4f}")
            logger.info(f"- Total candidates: {len(all_candidates)}")

if __name__ == "__main__":
    asyncio.run(main())