# src/dataset_collector.py

import json
import ast
import black
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import pytest
from io import StringIO
import sys
import logging
from datetime import datetime

@dataclass
class CodeExample:
    """Represents a single code example in the dataset"""
    requirement: str
    solution: str
    is_correct: bool
    test_cases: List[Dict[str, str]]  # List of test cases
    metadata: Dict  # Additional metadata like complexity, tags, etc.
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class DatasetCollector:
    """Collects and validates code examples for training"""
    
    def __init__(self, dataset_path: str = "golden_dataset.json"):
        self.dataset_path = Path(dataset_path)
        self.examples: List[CodeExample] = []
        self.load_existing_dataset()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_existing_dataset(self):
        """Load existing dataset if it exists"""
        if self.dataset_path.exists():
            try:
                with open(self.dataset_path, 'r') as f:
                    data = json.load(f)
                    self.examples = [CodeExample(**example) for example in data]
                self.logger.info(f"Loaded {len(self.examples)} existing examples")
            except Exception as e:
                self.logger.error(f"Error loading dataset: {e}")
                self.examples = []

    def save_dataset(self):
        """Save dataset to JSON file"""
        try:
            with open(self.dataset_path, 'w') as f:
                json.dump(
                    [asdict(example) for example in self.examples],
                    f,
                    indent=2
                )
            self.logger.info(f"Saved {len(self.examples)} examples to {self.dataset_path}")
        except Exception as e:
            self.logger.error(f"Error saving dataset: {e}")

    def validate_python_syntax(self, code: str) -> bool:
        """Check if code has valid Python syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def format_code(self, code: str) -> str:
        """Format code using black"""
        try:
            return black.format_str(code, mode=black.Mode())
        except:
            return code

    def run_test_cases(self, code: str, test_cases: List[Dict[str, str]]) -> bool:
        """Run test cases for the code"""
        # Create a test file
        test_code = [
            "import pytest",
            code,
            "\ndef test_solution():"
        ]
        
        # Add test cases
        for test in test_cases:
            test_code.append(f"    assert {test['input']} == {test['expected']}")
        
        test_file = "\n".join(test_code)
        
        # Run pytest
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            pytest.main(["-q", "--tb=no", "-s", "-c", test_file])
            return "FAILED" not in sys.stdout.getvalue()
        except:
            return False
        finally:
            sys.stdout = old_stdout

    def analyze_code_quality(self, code: str) -> Dict:
        """Analyze code quality metrics"""
        try:
            tree = ast.parse(code)
            
            # Count various metrics
            metrics = {
                "lines": len(code.splitlines()),
                "functions": len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                "classes": len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                "complexity": 0  # Basic complexity metric
            }
            
            # Simple complexity metric
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    metrics["complexity"] += 1
                    
            return metrics
        except:
            return {"error": "Could not analyze code"}

    def add_example(
        self,
        requirement: str,
        solution: str,
        test_cases: List[Dict[str, str]],
        tags: List[str] = None,
        auto_format: bool = True
    ) -> bool:
        """Add a new example to the dataset"""
        try:
            # Format code if requested
            if auto_format:
                solution = self.format_code(solution)
            
            # Validate syntax
            if not self.validate_python_syntax(solution):
                self.logger.error("Invalid Python syntax")
                return False
            
            # Run test cases
            if not self.run_test_cases(solution, test_cases):
                self.logger.error("Test cases failed")
                return False
            
            # Analyze code quality
            quality_metrics = self.analyze_code_quality(solution)
            
            # Create metadata
            metadata = {
                "tags": tags or [],
                "quality_metrics": quality_metrics,
                "validation_info": {
                    "syntax_valid": True,
                    "tests_passed": True,
                    "formatted": auto_format
                }
            }
            
            # Create example
            example = CodeExample(
                requirement=requirement,
                solution=solution,
                is_correct=True,
                test_cases=test_cases,
                metadata=metadata
            )
            
            # Add to dataset
            self.examples.append(example)
            self.save_dataset()
            
            self.logger.info("Successfully added new example")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding example: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            "total_examples": len(self.examples),
            "correct_examples": len([e for e in self.examples if e.is_correct]),
            "average_complexity": 0,
            "tags": {},
            "test_coverage": 0
        }
        
        # Calculate detailed statistics
        if self.examples:
            complexities = []
            all_tags = []
            test_cases = 0
            
            for example in self.examples:
                if "quality_metrics" in example.metadata:
                    complexities.append(
                        example.metadata["quality_metrics"].get("complexity", 0)
                    )
                all_tags.extend(example.metadata.get("tags", []))
                test_cases += len(example.test_cases)
            
            if complexities:
                stats["average_complexity"] = sum(complexities) / len(complexities)
            
            # Count tag frequencies
            for tag in set(all_tags):
                stats["tags"][tag] = all_tags.count(tag)
            
            stats["test_coverage"] = test_cases / len(self.examples)
        
        return stats

# Example usage
if __name__ == "__main__":
    collector = DatasetCollector()
    
    # Example: Adding a new code example
    example_requirement = "Write a function to check if a number is prime"
    example_solution = """
def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
    """
    
    test_cases = [
        {"input": "is_prime(7)", "expected": "True"},
        {"input": "is_prime(4)", "expected": "False"},
        {"input": "is_prime(2)", "expected": "True"},
        {"input": "is_prime(1)", "expected": "False"}
    ]
    
    collector.add_example(
        requirement=example_requirement,
        solution=example_solution,
        test_cases=test_cases,
        tags=["math", "basic", "algorithms"]
    )
    
    # Print statistics
    print(json.dumps(collector.get_statistics(), indent=2))