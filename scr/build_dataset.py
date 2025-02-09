# src/build_dataset.py

import click
from dataset_collector import DatasetCollector
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
import json
from typing import List, Dict

console = Console()

def display_statistics(stats: Dict):
    """Display dataset statistics in a nice table"""
    table = Table(title="Dataset Statistics")
    
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Examples", str(stats["total_examples"]))
    table.add_row("Correct Examples", str(stats["correct_examples"]))
    table.add_row("Average Complexity", f"{stats['average_complexity']:.2f}")
    table.add_row("Test Coverage", f"{stats['test_coverage']:.2f}")
    
    if stats["tags"]:
        tags_str = ", ".join(f"{k}({v})" for k, v in stats["tags"].items())
        table.add_row("Tags", tags_str)
    
    console.print(table)

def get_test_cases() -> List[Dict[str, str]]:
    """Interactively get test cases"""
    test_cases = []
    
    while True:
        console.print("\n[yellow]Add a test case:[/yellow]")
        input_expr = Prompt.ask("Input expression (e.g., func(5))")
        expected = Prompt.ask("Expected output")
        
        test_cases.append({
            "input": input_expr,
            "expected": expected
        })
        
        if not Confirm.ask("Add another test case?"):
            break
    
    return test_cases

@click.group()
def cli():
    """Tool for building a high-quality code dataset"""
    pass

@cli.command()
@click.option('--dataset', default='golden_dataset.json', help='Dataset file path')
def stats(dataset):
    """Show dataset statistics"""
    collector = DatasetCollector(dataset)
    stats = collector.get_statistics()
    display_statistics(stats)

@cli.command()
@click.option('--dataset', default='golden_dataset.json', help='Dataset file path')
def add(dataset):
    """Add a new example to the dataset"""
    collector = DatasetCollector(dataset)
    
    console.print("\n[bold cyan]Adding New Code Example[/bold cyan]")
    
    # Get requirement
    requirement = Prompt.ask("\nEnter the coding requirement")
    
    # Get solution
    console.print("\n[yellow]Enter the solution code (end with Ctrl+D or Ctrl+Z):[/yellow]")
    solution_lines = []
    try:
        while True:
            line = input()
            solution_lines.append(line)
    except EOFError:
        solution = "\n".join(solution_lines)
    
    # Get test cases
    test_cases = get_test_cases()
    
    # Get tags
    tags_str = Prompt.ask("\nEnter tags (comma-separated)")
    tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
    
    # Add example
    if collector.add_example(
        requirement=requirement,
        solution=solution,
        test_cases=test_cases,
        tags=tags
    ):
        console.print("\n[green]Successfully added example![/green]")
        
        # Show updated statistics
        stats = collector.get_statistics()
        display_statistics(stats)
    else:
        console.print("\n[red]Failed to add example. Please check the logs.[/red]")

@cli.command()
@click.option('--dataset', default='golden_dataset.json', help='Dataset file path')
def validate(dataset):
    """Validate all examples in the dataset"""
    collector = DatasetCollector(dataset)
    
    console.print("\n[bold cyan]Validating Dataset[/bold cyan]")
    
    for i, example in enumerate(collector.examples):
        console.print(f"\nChecking example {i+1}/{len(collector.examples)}")
        
        # Validate syntax
        if not collector.validate_python_syntax(example.solution):
            console.print(f"[red]Syntax error in example {i+1}[/red]")
            continue
        
        # Run tests
        if not collector.run_test_cases(example.solution, example.test_cases):
            console.print(f"[red]Tests failed for example {i+1}[/red]")
            continue
        
        console.print(f"[green]Example {i+1} is valid[/green]")
    
    console.print("\n[bold green]Validation complete![/bold green]")

if __name__ == "__main__":
    cli()