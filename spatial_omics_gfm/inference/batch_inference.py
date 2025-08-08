"""
Batch inference for large-scale spatial transcriptomics datasets.
Implements memory-efficient processing, parallel inference,
and streaming capabilities for population-scale analysis.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import h5py
import zarr
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc

from .efficient_inference import EfficientInference
from ..models.graph_transformer import SpatialGraphTransformer
from ..data.base import BaseSpatialDataset

logger = logging.getLogger(__name__)


class BatchInference(EfficientInference):
    """
    Large-scale batch inference for spatial transcriptomics datasets.
    
    Supports:
    - Memory-efficient chunked processing
    - Parallel processing across multiple GPUs/CPUs
    - Streaming inference for datasets too large for memory
    - Automatic memory management and optimization
    - Progress tracking and checkpointing
    """
    
    def __init__(
        self,
        model: SpatialGraphTransformer,
        batch_size: int = 32,
        max_memory_gb: float = 8.0,
        num_workers: int = 4,
        device: Optional[str] = None,
        use_mixed_precision: bool = True,
        chunk_overlap: int = 100,
        save_intermediate: bool = True,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Initialize batch inference engine.
        
        Args:
            model: Pre-trained spatial graph transformer
            batch_size: Batch size for inference
            max_memory_gb: Maximum memory usage in GB
            num_workers: Number of worker processes
            device: Device for inference
            use_mixed_precision: Whether to use mixed precision
            chunk_overlap: Overlap between chunks for boundary effects
            save_intermediate: Whether to save intermediate results
            checkpoint_dir: Directory for saving checkpoints
        """
        super().__init__(model, batch_size, use_mixed_precision, device)
        
        self.max_memory_gb = max_memory_gb
        self.num_workers = num_workers
        self.chunk_overlap = chunk_overlap
        self.save_intermediate = save_intermediate
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("./checkpoints")
        
        if self.save_intermediate:
            self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Memory monitoring
        self.memory_tracker = MemoryTracker(max_memory_gb)
        
        logger.info(f"Initialized BatchInference with max memory: {max_memory_gb}GB")
    
    def process_large_dataset(
        self,
        dataset_path: Union[str, Path],
        output_path: Union[str, Path],
        task_type: str = "embeddings",
        chunk_size: Optional[int] = None,
        resume_from_checkpoint: bool = True
    ) -> Dict[str, Any]:
        """
        Process large dataset in chunks.
        
        Args:
            dataset_path: Path to input dataset
            output_path: Path for output results
            task_type: Type of task ('embeddings', 'cell_types', 'interactions')
            chunk_size: Size of chunks (auto-determined if None)
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            Processing results and statistics
        """
        logger.info(f"Processing large dataset: {dataset_path}")
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = self._determine_optimal_chunk_size(dataset_path)
        
        # Check for existing checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{dataset_path.stem}.json"
        start_chunk = 0
        
        if resume_from_checkpoint and checkpoint_path.exists():
            start_chunk = self._load_checkpoint(checkpoint_path)
            logger.info(f"Resuming from chunk {start_chunk}")
        
        # Get dataset info
        dataset_info = self._get_dataset_info(dataset_path)
        total_cells = dataset_info['num_cells']
        num_chunks = (total_cells + chunk_size - 1) // chunk_size
        
        logger.info(f"Processing {total_cells} cells in {num_chunks} chunks of size {chunk_size}")
        
        # Process chunks
        results = []
        processing_stats = {
            'total_chunks': num_chunks,
            'processed_chunks': 0,
            'failed_chunks': [],
            'processing_times': []
        }
        
        with tqdm(total=num_chunks, desc="Processing chunks") as pbar:
            pbar.update(start_chunk)
            
            for chunk_idx in range(start_chunk, num_chunks):
                try:
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)
                    start_time.record()
                    
                    # Process chunk
                    chunk_result = self._process_chunk(
                        dataset_path, chunk_idx, chunk_size, task_type
                    )
                    
                    end_time.record()
                    torch.cuda.synchronize()
                    processing_time = start_time.elapsed_time(end_time) / 1000.0
                    
                    # Save chunk result
                    if self.save_intermediate:
                        chunk_output_path = output_path.parent / f"{output_path.stem}_chunk_{chunk_idx}.h5"
                        self._save_chunk_result(chunk_result, chunk_output_path)
                    
                    results.append(chunk_result)
                    processing_stats['processed_chunks'] += 1
                    processing_stats['processing_times'].append(processing_time)
                    
                    # Save checkpoint
                    self._save_checkpoint(checkpoint_path, chunk_idx + 1)
                    
                    # Memory cleanup
                    self.memory_tracker.cleanup_if_needed()
                    
                    pbar.update(1)
                    pbar.set_postfix({
                        'Memory': f"{self.memory_tracker.get_memory_usage():.1f}GB",
                        'Time': f"{processing_time:.2f}s"
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to process chunk {chunk_idx}: {e}")
                    processing_stats['failed_chunks'].append(chunk_idx)
                    continue
        
        # Combine results
        logger.info("Combining chunk results")
        final_results = self._combine_chunk_results(results, output_path)
        
        # Clean up intermediate files if requested
        if not self.save_intermediate:
            self._cleanup_intermediate_files(output_path.parent, output_path.stem)
        
        processing_stats['mean_processing_time'] = np.mean(processing_stats['processing_times'])
        processing_stats['total_processing_time'] = sum(processing_stats['processing_times'])
        
        return {
            'results': final_results,
            'statistics': processing_stats
        }
    
    def _determine_optimal_chunk_size(self, dataset_path: Path) -> int:
        """Determine optimal chunk size based on available memory."""
        logger.info("Determining optimal chunk size")
        
        # Get dataset info
        dataset_info = self._get_dataset_info(dataset_path)
        
        # Estimate memory usage per cell
        num_genes = dataset_info['num_genes']
        embedding_dim = self.model.config.hidden_dim
        
        # Memory per cell (in bytes)
        # Expression matrix + embeddings + intermediate tensors
        memory_per_cell = (
            num_genes * 4 +  # Expression (float32)
            embedding_dim * 4 +  # Embeddings (float32)
            embedding_dim * 8  # Intermediate tensors (safety factor)
        )
        
        # Available memory in bytes
        available_memory = self.max_memory_gb * 1024**3 * 0.8  # Use 80% of max memory
        
        # Calculate chunk size
        chunk_size = int(available_memory / memory_per_cell)
        
        # Apply constraints
        chunk_size = max(chunk_size, 100)  # Minimum chunk size
        chunk_size = min(chunk_size, 50000)  # Maximum chunk size
        
        logger.info(f"Determined optimal chunk size: {chunk_size}")
        return chunk_size
    
    def _get_dataset_info(self, dataset_path: Path) -> Dict[str, Any]:
        """Get basic information about the dataset."""
        try:
            if dataset_path.suffix == '.h5':
                with h5py.File(dataset_path, 'r') as f:
                    if 'X' in f:
                        shape = f['X'].shape
                    elif 'matrix' in f:
                        shape = f['matrix'].shape
                    else:
                        # Explore structure
                        for key in f.keys():
                            if hasattr(f[key], 'shape') and len(f[key].shape) == 2:
                                shape = f[key].shape
                                break
                        else:
                            raise ValueError("Could not find expression matrix in H5 file")
                    
                    return {
                        'num_cells': shape[0],
                        'num_genes': shape[1],
                        'format': 'h5'
                    }
            
            elif dataset_path.suffix == '.zarr':
                store = zarr.open(dataset_path, mode='r')
                shape = store['X'].shape
                return {
                    'num_cells': shape[0],
                    'num_genes': shape[1],
                    'format': 'zarr'
                }
            
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path.suffix}")
                
        except Exception as e:
            logger.error(f"Failed to get dataset info: {e}")
            # Return default values
            return {
                'num_cells': 10000,
                'num_genes': 2000,
                'format': 'unknown'
            }
    
    def _process_chunk(
        self,
        dataset_path: Path,
        chunk_idx: int,
        chunk_size: int,
        task_type: str
    ) -> Dict[str, Any]:
        """Process a single chunk of data."""
        # Load chunk data
        chunk_data = self._load_chunk(dataset_path, chunk_idx, chunk_size)
        
        # Create temporary dataset object
        chunk_dataset = ChunkDataset(chunk_data)
        
        # Process with model
        if task_type == "embeddings":
            results = self._compute_embeddings_chunk(chunk_dataset)
        elif task_type == "cell_types":
            results = self._predict_cell_types_chunk(chunk_dataset)
        elif task_type == "interactions":
            results = self._predict_interactions_chunk(chunk_dataset)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
        
        # Add chunk metadata
        results['chunk_info'] = {
            'chunk_idx': chunk_idx,
            'chunk_size': len(chunk_data['expression']),
            'start_idx': chunk_idx * chunk_size,
            'end_idx': min((chunk_idx + 1) * chunk_size, chunk_data.get('total_cells', chunk_size))
        }
        
        return results
    
    def _load_chunk(self, dataset_path: Path, chunk_idx: int, chunk_size: int) -> Dict[str, Any]:
        """Load a chunk of data from the dataset."""
        start_idx = chunk_idx * chunk_size
        end_idx = (chunk_idx + 1) * chunk_size
        
        if dataset_path.suffix == '.h5':
            return self._load_h5_chunk(dataset_path, start_idx, end_idx)
        elif dataset_path.suffix == '.zarr':
            return self._load_zarr_chunk(dataset_path, start_idx, end_idx)
        else:
            raise ValueError(f"Unsupported format: {dataset_path.suffix}")
    
    def _load_h5_chunk(self, dataset_path: Path, start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Load chunk from H5 file."""
        with h5py.File(dataset_path, 'r') as f:
            # Load expression data
            if 'X' in f:
                expression = f['X'][start_idx:end_idx]
            else:
                # Find expression matrix
                for key in f.keys():
                    if hasattr(f[key], 'shape') and len(f[key].shape) == 2:
                        expression = f[key][start_idx:end_idx]
                        break
                else:
                    raise ValueError("No expression matrix found")
            
            # Load spatial coordinates if available
            spatial_coords = None
            if 'obsm' in f and 'spatial' in f['obsm']:
                spatial_coords = f['obsm']['spatial'][start_idx:end_idx]
            elif 'spatial' in f:
                spatial_coords = f['spatial'][start_idx:end_idx]
            
            # Load gene names if available
            gene_names = None
            if 'var' in f and 'gene_ids' in f['var']:
                gene_names = f['var']['gene_ids'][:]
            elif 'gene_names' in f:
                gene_names = f['gene_names'][:]
            
            # Load cell metadata if available
            cell_metadata = {}
            if 'obs' in f:
                for key in f['obs'].keys():
                    cell_metadata[key] = f['obs'][key][start_idx:end_idx]
        
        return {
            'expression': expression,
            'spatial_coords': spatial_coords,
            'gene_names': gene_names,
            'cell_metadata': cell_metadata,
            'chunk_start': start_idx,
            'chunk_end': end_idx
        }
    
    def _load_zarr_chunk(self, dataset_path: Path, start_idx: int, end_idx: int) -> Dict[str, Any]:
        """Load chunk from Zarr store."""
        store = zarr.open(dataset_path, mode='r')
        
        # Load expression data
        expression = store['X'][start_idx:end_idx]
        
        # Load spatial coordinates if available
        spatial_coords = None
        if 'spatial' in store:
            spatial_coords = store['spatial'][start_idx:end_idx]
        
        # Load gene names if available
        gene_names = None
        if 'gene_names' in store:
            gene_names = store['gene_names'][:]
        
        return {
            'expression': expression,
            'spatial_coords': spatial_coords,
            'gene_names': gene_names,
            'chunk_start': start_idx,
            'chunk_end': end_idx
        }
    
    def _compute_embeddings_chunk(self, chunk_dataset: 'ChunkDataset') -> Dict[str, Any]:
        """Compute embeddings for a chunk."""
        dataloader = DataLoader(
            chunk_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0  # Use 0 for GPU inference
        )
        
        embeddings = []
        
        with torch.no_grad():
            for batch in dataloader:
                gene_expression = batch['expression'].to(self.device)
                spatial_coords = batch['spatial_coords'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                edge_attr = batch['edge_attr'].to(self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        batch_embeddings = self.model.encode(
                            gene_expression, spatial_coords, edge_index, edge_attr
                        )
                else:
                    batch_embeddings = self.model.encode(
                        gene_expression, spatial_coords, edge_index, edge_attr
                    )
                
                embeddings.append(batch_embeddings.cpu())
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings, dim=0)
        
        return {
            'embeddings': all_embeddings.numpy(),
            'task_type': 'embeddings'
        }
    
    def _predict_cell_types_chunk(self, chunk_dataset: 'ChunkDataset') -> Dict[str, Any]:
        """Predict cell types for a chunk."""
        # First get embeddings
        embeddings_result = self._compute_embeddings_chunk(chunk_dataset)
        embeddings = torch.tensor(embeddings_result['embeddings'])
        
        # Apply cell type prediction head if available
        if hasattr(self.model, 'cell_type_head'):
            with torch.no_grad():
                cell_type_logits = self.model.cell_type_head(embeddings.to(self.device))
                cell_type_probs = torch.softmax(cell_type_logits, dim=-1)
                predicted_types = torch.argmax(cell_type_probs, dim=-1)
                
                return {
                    'cell_type_predictions': predicted_types.cpu().numpy(),
                    'cell_type_probabilities': cell_type_probs.cpu().numpy(),
                    'embeddings': embeddings.numpy(),
                    'task_type': 'cell_types'
                }
        else:
            logger.warning("No cell type head available, returning embeddings only")
            return embeddings_result
    
    def _predict_interactions_chunk(self, chunk_dataset: 'ChunkDataset') -> Dict[str, Any]:
        """Predict interactions for a chunk."""
        # First get embeddings
        embeddings_result = self._compute_embeddings_chunk(chunk_dataset)
        embeddings = torch.tensor(embeddings_result['embeddings'])
        
        # Apply interaction prediction head if available
        if hasattr(self.model, 'interaction_head'):
            # Need to reconstruct edge information for interactions
            dataloader = DataLoader(chunk_dataset, batch_size=1, shuffle=False)
            
            interactions = []
            
            with torch.no_grad():
                for batch in dataloader:
                    edge_index = batch['edge_index'].to(self.device)
                    edge_attr = batch['edge_attr'].to(self.device)
                    
                    batch_start = batch['batch_start'].item()
                    batch_end = batch['batch_end'].item()
                    
                    batch_embeddings = embeddings[batch_start:batch_end].to(self.device)
                    
                    interaction_logits = self.model.interaction_head(
                        batch_embeddings, edge_index, edge_attr
                    )
                    
                    interaction_probs = torch.softmax(interaction_logits, dim=-1)
                    interactions.append(interaction_probs.cpu())
            
            all_interactions = torch.cat(interactions, dim=0)
            
            return {
                'interaction_predictions': all_interactions.numpy(),
                'embeddings': embeddings.numpy(),
                'task_type': 'interactions'
            }
        else:
            logger.warning("No interaction head available, returning embeddings only")
            return embeddings_result
    
    def _save_chunk_result(self, result: Dict[str, Any], output_path: Path) -> None:
        """Save chunk result to file."""
        import h5py
        
        with h5py.File(output_path, 'w') as f:
            for key, value in result.items():
                if key == 'chunk_info':
                    # Save metadata as attributes
                    for info_key, info_value in value.items():
                        f.attrs[info_key] = info_value
                elif isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip')
                elif isinstance(value, str):
                    f.attrs[key] = value
    
    def _combine_chunk_results(self, results: List[Dict[str, Any]], output_path: Path) -> Dict[str, Any]:
        """Combine results from all chunks."""
        if not results:
            return {}
        
        combined = {}
        task_type = results[0].get('task_type', 'embeddings')
        
        # Combine arrays
        for key in results[0].keys():
            if key in ['chunk_info', 'task_type']:
                continue
            
            if isinstance(results[0][key], np.ndarray):
                arrays = [result[key] for result in results if key in result]
                combined[key] = np.concatenate(arrays, axis=0)
        
        # Save combined results
        self._save_combined_results(combined, output_path, task_type)
        
        return combined
    
    def _save_combined_results(self, results: Dict[str, Any], output_path: Path, task_type: str) -> None:
        """Save combined results to output file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as f:
            f.attrs['task_type'] = task_type
            f.attrs['num_cells'] = len(list(results.values())[0]) if results else 0
            
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value, compression='gzip')
        
        logger.info(f"Saved combined results to {output_path}")
    
    def _save_checkpoint(self, checkpoint_path: Path, chunk_idx: int) -> None:
        """Save processing checkpoint."""
        import json
        
        checkpoint_data = {
            'last_processed_chunk': chunk_idx,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f)
    
    def _load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load processing checkpoint."""
        import json
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data.get('last_processed_chunk', 0)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return 0
    
    def _cleanup_intermediate_files(self, directory: Path, stem: str) -> None:
        """Clean up intermediate chunk files."""
        chunk_files = list(directory.glob(f"{stem}_chunk_*.h5"))
        for file_path in chunk_files:
            try:
                file_path.unlink()
                logger.debug(f"Removed intermediate file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
    
    def stream_predictions(
        self,
        dataset_path: Union[str, Path],
        task_type: str = "embeddings",
        chunk_size: Optional[int] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream predictions for very large datasets.
        
        Args:
            dataset_path: Path to input dataset
            task_type: Type of task to perform
            chunk_size: Size of chunks for streaming
            
        Yields:
            Prediction results for each chunk
        """
        logger.info(f"Starting streaming inference for {dataset_path}")
        
        dataset_path = Path(dataset_path)
        
        if chunk_size is None:
            chunk_size = self._determine_optimal_chunk_size(dataset_path)
        
        dataset_info = self._get_dataset_info(dataset_path)
        total_cells = dataset_info['num_cells']
        num_chunks = (total_cells + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            try:
                # Process chunk
                chunk_result = self._process_chunk(
                    dataset_path, chunk_idx, chunk_size, task_type
                )
                
                # Clean up memory
                self.memory_tracker.cleanup_if_needed()
                
                yield chunk_result
                
            except Exception as e:
                logger.error(f"Failed to process chunk {chunk_idx}: {e}")
                continue


class ChunkDataset(Dataset):
    """Dataset wrapper for chunk processing."""
    
    def __init__(self, chunk_data: Dict[str, Any]):
        self.chunk_data = chunk_data
        self.expression = chunk_data['expression']
        self.spatial_coords = chunk_data.get('spatial_coords')
        
        # Build spatial graph for the chunk
        if self.spatial_coords is not None:
            self.edge_index, self.edge_attr = self._build_spatial_graph()
        else:
            # Create dummy graph
            n_cells = len(self.expression)
            self.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            self.edge_attr = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    
    def _build_spatial_graph(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build spatial graph for the chunk."""
        from sklearn.neighbors import NearestNeighbors
        
        coords = self.spatial_coords
        n_neighbors = min(6, len(coords) - 1)
        
        if n_neighbors <= 0:
            # Single cell case
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Build edge list
        edges = []
        edge_features = []
        
        for i in range(len(coords)):
            for j in range(1, len(indices[i])):  # Skip self (index 0)
                neighbor_idx = indices[i][j]
                distance = distances[i][j]
                
                # Calculate direction
                dx = coords[neighbor_idx, 0] - coords[i, 0]
                dy = coords[neighbor_idx, 1] - coords[i, 1]
                
                edges.append([i, neighbor_idx])
                edge_features.append([distance, dx, dy])
        
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
        
        return edge_index, edge_attr
    
    def __len__(self) -> int:
        return 1  # Each chunk is processed as a single batch
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        expression = torch.tensor(self.expression, dtype=torch.float32)
        spatial_coords = torch.tensor(self.spatial_coords, dtype=torch.float32) if self.spatial_coords is not None else torch.zeros((len(expression), 2))
        
        return {
            'expression': expression,
            'spatial_coords': spatial_coords,
            'edge_index': self.edge_index,
            'edge_attr': self.edge_attr,
            'batch_start': 0,
            'batch_end': len(expression)
        }


class MemoryTracker:
    """Utility class for tracking and managing memory usage."""
    
    def __init__(self, max_memory_gb: float):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = max_memory_gb * 1024**3
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024**3
    
    def cleanup_if_needed(self) -> None:
        """Clean up memory if usage is too high."""
        current_usage = self.get_memory_usage()
        
        if current_usage > self.max_memory_gb * 0.8:  # 80% threshold
            logger.debug(f"Memory usage high ({current_usage:.1f}GB), cleaning up")
            
            # Force garbage collection
            gc.collect()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            new_usage = self.get_memory_usage()
            logger.debug(f"Memory usage after cleanup: {new_usage:.1f}GB")
    
    def is_memory_available(self, required_gb: float) -> bool:
        """Check if required memory is available."""
        current_usage = self.get_memory_usage()
        return (current_usage + required_gb) <= self.max_memory_gb


def process_dataset_parallel(
    dataset_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    model: SpatialGraphTransformer,
    task_type: str = "embeddings",
    max_workers: int = 4,
    **inference_kwargs
) -> Dict[str, Any]:
    """
    Process multiple datasets in parallel.
    
    Args:
        dataset_paths: List of dataset paths to process
        output_dir: Output directory for results
        model: Pre-trained model
        task_type: Type of task to perform
        max_workers: Maximum number of parallel workers
        **inference_kwargs: Additional arguments for BatchInference
        
    Returns:
        Processing results for all datasets
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    def process_single_dataset(dataset_path: Path) -> Dict[str, Any]:
        """Process a single dataset."""
        try:
            inference_engine = BatchInference(model, **inference_kwargs)
            
            output_path = output_dir / f"{dataset_path.stem}_results.h5"
            
            result = inference_engine.process_large_dataset(
                dataset_path, output_path, task_type
            )
            
            return {
                'dataset_path': str(dataset_path),
                'output_path': str(output_path),
                'success': True,
                'statistics': result['statistics']
            }
            
        except Exception as e:
            logger.error(f"Failed to process {dataset_path}: {e}")
            return {
                'dataset_path': str(dataset_path),
                'success': False,
                'error': str(e)
            }
    
    # Process datasets in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_dataset, dataset_paths))
    
    # Combine results
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    logger.info(f"Processed {len(successful)}/{len(dataset_paths)} datasets successfully")
    
    return {
        'successful_datasets': successful,
        'failed_datasets': failed,
        'total_datasets': len(dataset_paths),
        'success_rate': len(successful) / len(dataset_paths)
    }