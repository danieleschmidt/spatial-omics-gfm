"""
Command-line interface for Spatial-Omics GFM.

This module provides a comprehensive CLI for all major functionality
including data loading, model training, inference, and visualization.
"""

import typer
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List
import sys
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import warnings

# Import core modules
from .models import SpatialGraphTransformer, load_pretrained_model, AVAILABLE_MODELS
from .data import VisiumDataset, SlideSeqDataset, XeniumDataset, MERFISHDataset
from .tasks import CellTypeClassifier, InteractionPredictor, PathwayAnalyzer
from .training import FineTuner
from .inference import EfficientInference

app = typer.Typer(help="Spatial-Omics GFM: Graph Foundation Model for Spatial Transcriptomics")
console = Console()

# Subcommands
model_app = typer.Typer(help="Model management commands")
data_app = typer.Typer(help="Data processing commands")
train_app = typer.Typer(help="Training commands")
predict_app = typer.Typer(help="Prediction commands")
viz_app = typer.Typer(help="Visualization commands")

app.add_typer(model_app, name="model")
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(predict_app, name="predict")
app.add_typer(viz_app, name="viz")


@app.command()
def info():
    """Display information about Spatial-Omics GFM."""
    console.print("[bold blue]Spatial-Omics GFM[/bold blue]")
    console.print("Graph Foundation Model for Spatial Transcriptomics")
    console.print()
    
    # System info
    console.print(f"PyTorch version: {torch.__version__}")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"CUDA devices: {torch.cuda.device_count()}")
    console.print()
    
    # Available models
    table = Table(title="Available Pre-trained Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Parameters", style="magenta")
    table.add_column("Description", style="green")
    
    for name, info in AVAILABLE_MODELS.items():
        table.add_row(name, info["parameters"], info["description"])
    
    console.print(table)


@model_app.command("list")
def list_models():
    """List all available pre-trained models."""
    table = Table(title="Available Pre-trained Models")
    table.add_column("Model Name", style="cyan")
    table.add_column("Parameters", style="magenta")
    table.add_column("Training Data", style="yellow")
    table.add_column("Tasks", style="green")
    
    for name, info in AVAILABLE_MODELS.items():
        tasks = ", ".join(info["tasks"])
        table.add_row(name, info["parameters"], info["training_data"], tasks)
    
    console.print(table)


@model_app.command("download")
def download_model(
    model_name: str = typer.Argument(help="Name of the model to download"),
    cache_dir: Optional[str] = typer.Option(None, help="Cache directory"),
    force: bool = typer.Option(False, help="Force re-download")
):
    """Download a pre-trained model."""
    if model_name not in AVAILABLE_MODELS:
        console.print(f"[red]Error: Model '{model_name}' not found[/red]")
        available = ", ".join(AVAILABLE_MODELS.keys())
        console.print(f"Available models: {available}")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Downloading {model_name}...", total=None)
        
        try:
            model = load_pretrained_model(
                model_name,
                cache_dir=cache_dir,
                force_download=force
            )
            progress.update(task, description=f"✓ Downloaded {model_name}")
            console.print(f"[green]Successfully downloaded {model_name}[/green]")
            
        except Exception as e:
            console.print(f"[red]Error downloading model: {e}[/red]")
            raise typer.Exit(1)


@data_app.command("process")
def process_data(
    platform: str = typer.Argument(help="Platform: visium, slideseq, xenium, merfish"),
    data_path: str = typer.Argument(help="Path to data files"),
    output_path: str = typer.Argument(help="Output path for processed data"),
    normalize: bool = typer.Option(True, help="Apply normalization"),
    filter_genes: bool = typer.Option(True, help="Filter genes"),
    min_cells: int = typer.Option(10, help="Minimum cells per gene"),
    min_genes: int = typer.Option(200, help="Minimum genes per cell"),
    hvg: int = typer.Option(3000, help="Number of highly variable genes")
):
    """Process spatial transcriptomics data."""
    platform = platform.lower()
    
    # Select appropriate dataset class
    dataset_classes = {
        "visium": VisiumDataset,
        "slideseq": SlideSeqDataset,
        "xenium": XeniumDataset,
        "merfish": MERFISHDataset
    }
    
    if platform not in dataset_classes:
        console.print(f"[red]Error: Unsupported platform '{platform}'[/red]")
        console.print(f"Supported platforms: {', '.join(dataset_classes.keys())}")
        raise typer.Exit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing data...", total=None)
        
        try:
            # Create dataset configuration
            from .data.base import SpatialDataConfig
            config = SpatialDataConfig(
                normalize=normalize,
                filter_genes=filter_genes,
                min_cells_per_gene=min_cells,
                min_genes_per_cell=min_genes,
                highly_variable_genes=hvg
            )
            
            # Initialize dataset
            dataset_class = dataset_classes[platform]
            dataset = dataset_class(config=config)
            
            # Process data
            progress.update(task, description="Loading data...")
            dataset.setup(data_path)
            
            progress.update(task, description="Saving processed data...")
            dataset.save(output_path)
            
            console.print(f"[green]Successfully processed {platform} data[/green]")
            console.print(f"Cells: {dataset.num_cells}, Genes: {dataset.num_genes}")
            console.print(f"Output saved to: {output_path}")
            
        except Exception as e:
            console.print(f"[red]Error processing data: {e}[/red]")
            raise typer.Exit(1)


@predict_app.command("cell-types")
def predict_cell_types(
    model_name: str = typer.Argument(help="Pre-trained model name"),
    data_path: str = typer.Argument(help="Path to processed data"),
    output_path: str = typer.Argument(help="Output path for predictions"),
    device: str = typer.Option("auto", help="Device: cuda, cpu, or auto"),
    batch_size: int = typer.Option(32, help="Batch size for inference"),
    confidence_threshold: float = typer.Option(0.5, help="Confidence threshold")
):
    """Predict cell types using a pre-trained model."""
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Predicting cell types...", total=None)
        
        try:
            # Load model
            progress.update(task, description="Loading model...")
            model = load_pretrained_model(model_name, device=device)
            
            # Load data
            progress.update(task, description="Loading data...")
            from .data.base import BaseSpatialDataset
            dataset = BaseSpatialDataset.load(data_path)
            
            # Setup inference engine
            progress.update(task, description="Setting up inference...")
            inference_engine = EfficientInference(
                model=model,
                batch_size=batch_size,
                device=device
            )
            
            # Run predictions
            progress.update(task, description="Running predictions...")
            predictions = inference_engine.predict_cell_types(
                dataset,
                confidence_threshold=confidence_threshold
            )
            
            # Save results
            progress.update(task, description="Saving results...")
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save predictions as JSON
            results = {
                "model_name": model_name,
                "num_cells": len(predictions["predictions"]),
                "confidence_threshold": confidence_threshold,
                "predictions": predictions["predictions"].tolist(),
                "probabilities": predictions["probabilities"].tolist(),
                "confidence": predictions["confidence"].tolist(),
                "cell_type_names": predictions.get("cell_type_names", [])
            }
            
            with open(output_path / "cell_type_predictions.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            console.print(f"[green]Successfully predicted cell types[/green]")
            console.print(f"Total cells: {len(predictions['predictions'])}")
            console.print(f"High confidence predictions: {predictions['high_confidence_mask'].sum().item()}")
            console.print(f"Results saved to: {output_path}")
            
        except Exception as e:
            console.print(f"[red]Error predicting cell types: {e}[/red]")
            raise typer.Exit(1)


@predict_app.command("interactions")
def predict_interactions(
    model_name: str = typer.Argument(help="Pre-trained model name"),
    data_path: str = typer.Argument(help="Path to processed data"),
    output_path: str = typer.Argument(help="Output path for predictions"),
    device: str = typer.Option("auto", help="Device: cuda, cpu, or auto"),
    distance_threshold: float = typer.Option(200.0, help="Distance threshold (μm)"),
    confidence_threshold: float = typer.Option(0.8, help="Confidence threshold")
):
    """Predict cell-cell interactions."""
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Predicting interactions...", total=None)
        
        try:
            # Load model
            progress.update(task, description="Loading model...")
            model = load_pretrained_model(model_name, device=device)
            
            # Load data
            progress.update(task, description="Loading data...")
            from .data.base import BaseSpatialDataset
            dataset = BaseSpatialDataset.load(data_path)
            
            # Setup interaction predictor
            progress.update(task, description="Setting up predictor...")
            predictor = InteractionPredictor(
                model=model,
                distance_threshold=distance_threshold
            )
            
            # Run predictions
            progress.update(task, description="Running predictions...")
            interactions = predictor.predict_interactions(
                dataset,
                confidence_threshold=confidence_threshold
            )
            
            # Save results
            progress.update(task, description="Saving results...")
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            results = {
                "model_name": model_name,
                "distance_threshold": distance_threshold,
                "confidence_threshold": confidence_threshold,
                "num_interactions": len(interactions["interactions"]),
                "interactions": interactions
            }
            
            with open(output_path / "interaction_predictions.json", 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"[green]Successfully predicted interactions[/green]")
            console.print(f"Total interactions: {len(interactions['interactions'])}")
            console.print(f"Results saved to: {output_path}")
            
        except Exception as e:
            console.print(f"[red]Error predicting interactions: {e}[/red]")
            raise typer.Exit(1)


@train_app.command("finetune")
def finetune_model(
    model_name: str = typer.Argument(help="Base model to fine-tune"),
    train_data: str = typer.Argument(help="Path to training data"),
    output_dir: str = typer.Argument(help="Output directory for fine-tuned model"),
    task: str = typer.Option("cell_typing", help="Task: cell_typing, interactions, pathways"),
    epochs: int = typer.Option(10, help="Number of training epochs"),
    learning_rate: float = typer.Option(1e-5, help="Learning rate"),
    batch_size: int = typer.Option(4, help="Batch size"),
    device: str = typer.Option("auto", help="Device: cuda, cpu, or auto")
):
    """Fine-tune a pre-trained model on custom data."""
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task_id = progress.add_task("Fine-tuning model...", total=None)
        
        try:
            # Load base model
            progress.update(task_id, description="Loading base model...")
            model = load_pretrained_model(model_name, device=device)
            
            # Load training data
            progress.update(task_id, description="Loading training data...")
            from .data.base import BaseSpatialDataset
            dataset = BaseSpatialDataset.load(train_data)
            
            # Setup fine-tuner
            progress.update(task_id, description="Setting up training...")
            fine_tuner = FineTuner(
                base_model=model,
                task=task,
                learning_rate=learning_rate,
                device=device
            )
            
            # Run fine-tuning
            progress.update(task_id, description="Training...")
            fine_tuned_model = fine_tuner.fine_tune(
                dataset,
                epochs=epochs,
                batch_size=batch_size,
                output_dir=output_dir
            )
            
            console.print(f"[green]Successfully fine-tuned model[/green]")
            console.print(f"Model saved to: {output_dir}")
            
        except Exception as e:
            console.print(f"[red]Error fine-tuning model: {e}[/red]")
            raise typer.Exit(1)


@viz_app.command("spatial")
def visualize_spatial(
    data_path: str = typer.Argument(help="Path to processed data"),
    predictions_path: Optional[str] = typer.Option(None, help="Path to predictions"),
    output_path: str = typer.Argument(help="Output path for visualization"),
    color_by: str = typer.Option("cell_type", help="Color by: cell_type, gene, confidence"),
    interactive: bool = typer.Option(True, help="Create interactive plot"),
    width: int = typer.Option(800, help="Plot width"),
    height: int = typer.Option(600, help="Plot height")
):
    """Create spatial visualization of data and predictions."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Creating visualization...", total=None)
        
        try:
            # Load data
            progress.update(task, description="Loading data...")
            from .data.base import BaseSpatialDataset
            dataset = BaseSpatialDataset.load(data_path)
            
            # Load predictions if provided
            predictions = None
            if predictions_path:
                progress.update(task, description="Loading predictions...")
                with open(predictions_path, 'r') as f:
                    predictions = json.load(f)
            
            # Create visualization
            progress.update(task, description="Generating plot...")
            from .visualization import SpatialPlotter
            
            plotter = SpatialPlotter()
            
            if interactive:
                fig = plotter.plot_interactive_spatial(
                    dataset,
                    predictions=predictions,
                    color_by=color_by,
                    width=width,
                    height=height
                )
                fig.write_html(f"{output_path}/spatial_plot.html")
            else:
                fig = plotter.plot_spatial(
                    dataset,
                    predictions=predictions,
                    color_by=color_by,
                    figsize=(width/100, height/100)
                )
                fig.savefig(f"{output_path}/spatial_plot.png", dpi=300, bbox_inches='tight')
            
            console.print(f"[green]Successfully created visualization[/green]")
            console.print(f"Saved to: {output_path}")
            
        except Exception as e:
            console.print(f"[red]Error creating visualization: {e}[/red]")
            raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    from . import __version__
    console.print(f"Spatial-Omics GFM version {__version__}")


def main():
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()