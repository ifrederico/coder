# my_ai_coder_ebm_advanced

A more advanced “self-coding AI” scaffold that:
- Uses an **Energy-Based Model (EBM)** to score (code, requirement) pairs
- Employs **margin-based training** with **hard-negative sampling**
- Iterates: generate code, test it, collect positives & negatives, train EBM, repeat

## Quick Start

1. **Create a virtual env**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate

2. **Install dependencies**:
   pip install -r requirements.txt

3. **Run**:
python src/main.py

4. Customize:
Integrate a real embedding model in embedder.py
Replace test_runner.py with your actual test harness
Expand the “hard negative” logic in advanced_neg_sampling.py





my_ai_coder_ebm_advanced/
├── README.md
├── requirements.txt
└── src
    ├── main.py
    ├── code_llm.py
    ├── ebm_model.py
    ├── ebm_trainer.py
    ├── embedder.py
    ├── memory_store.py
    ├── self_coding_agent.py
    ├── test_runner.py
    └── advanced_neg_sampling.py
