# src/main.py

import asyncio
import torch
import logging
from code_llm import CodeLLM
from ebm_model import EnergyModel
from embedder import CodeRequirementEmbedder
from memory_store import MemoryStore
from self_coding_agent import EnhancedSelfCodingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    # Use CPU for stability
    device = 'cpu'
    logger.info(f"Using device: {device}")

    try:
        # Initialize components with enhanced versions
        llm = CodeLLM(
            model_name="microsoft/phi-1_5",  # More memory efficient model
            device=device
        )
        
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
            num_candidates=3,  # Reduced for memory efficiency
            max_workers=2  # Reduced for stability
        ) as agent:
            
            # Start with a simple task
            tasks = [
                {
                    "requirement": "Write a function to check if a string is a palindrome",
                    "template": """
                    Write a simple Python function that checks if a string is a palindrome.
                    The function should:
                    1. Take a string as input
                    2. Return True if it's a palindrome, False otherwise
                    3. Ignore spaces and case
                    4. Include type hints
                    Only return the code, no explanations.
                    """
                }
            ]

            for task in tasks:
                logger.info(f"\n=== Starting task: {task['requirement']} ===")
                
                try:
                    result = await agent.solve_task(
                        requirement=task['requirement'],
                        prompt_template=task['template'],
                        max_iterations=2  # Reduced for testing
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
                    if all_candidates:
                        pass_rate = sum(1 for c in all_candidates if c.passed_tests) / len(all_candidates)
                        avg_energy = sum(c.energy for c in all_candidates) / len(all_candidates)
                        
                        logger.info(f"Statistics:")
                        logger.info(f"- Pass rate: {pass_rate:.2%}")
                        logger.info(f"- Average energy: {avg_energy:.4f}")
                        logger.info(f"- Total candidates: {len(all_candidates)}")
                
                except Exception as e:
                    logger.error(f"Error processing task: {str(e)}")
                    continue

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())