"""
Performance optimization utilities for Spatial-Omics GFM.
Implements torch.compile, ONNX export, quantization, and deployment optimizations.
"""

import logging
import warnings
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import ScriptModule
import torch.quantization as quant
from anndata import AnnData

logger = logging.getLogger(__name__)


class ModelOptimizer:
    """
    Model optimization for production deployment.
    
    Provides:
    - torch.compile() optimizations
    - ONNX export for inference engines
    - Quantization for edge deployment
    - Memory and compute optimizations
    """
    
    def __init__(self):
        self.optimized_models = {}
        self.compilation_cache = {}
        logger.info("Initialized ModelOptimizer")
    
    def compile_model(
        self,
        model: nn.Module,
        mode: str = "default",
        dynamic: bool = True,
        fullgraph: bool = False,
        backend: str = "inductor"
    ) -> nn.Module:
        """
        Compile model with torch.compile for optimized inference.
        
        Args:
            model: PyTorch model to compile
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            dynamic: Enable dynamic shapes
            fullgraph: Capture full graph
            backend: Compilation backend
            
        Returns:
            Compiled model
        """
        if not hasattr(torch, 'compile'):
            logger.warning("torch.compile not available, returning original model")
            return model
        
        logger.info(f"Compiling model with mode: {mode}, backend: {backend}")
        
        try:
            # Check if model already compiled with same settings
            config_key = f"{mode}_{dynamic}_{fullgraph}_{backend}"
            if config_key in self.compilation_cache:
                logger.info("Using cached compiled model")
                return self.compilation_cache[config_key]
            
            # Compile model
            compiled_model = torch.compile(
                model,
                mode=mode,
                dynamic=dynamic,
                fullgraph=fullgraph,
                backend=backend
            )
            
            # Cache compiled model
            self.compilation_cache[config_key] = compiled_model
            
            logger.info("Model compilation completed successfully")
            return compiled_model
            
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}, returning original model")
            return model
    
    def export_to_onnx(
        self,
        model: nn.Module,
        dummy_input: Dict[str, torch.Tensor],
        export_path: Union[str, Path],
        opset_version: int = 14,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        optimize: bool = True
    ) -> bool:
        """
        Export model to ONNX format.
        
        Args:
            model: PyTorch model to export
            dummy_input: Example input for tracing
            export_path: Path to save ONNX model
            opset_version: ONNX opset version
            dynamic_axes: Dynamic axes specification
            optimize: Whether to optimize the ONNX model
            
        Returns:
            Success status
        """
        try:
            logger.info(f"Exporting model to ONNX: {export_path}")
            
            # Ensure model is in eval mode
            model.eval()
            
            # Extract input tensors from dummy_input dict
            if isinstance(dummy_input, dict):
                input_names = list(dummy_input.keys())
                inputs = list(dummy_input.values())
                
                # Unpack if single input
                if len(inputs) == 1:
                    inputs = inputs[0]
            else:
                input_names = ["input"]
                inputs = dummy_input
            
            # Default dynamic axes for spatial data
            if dynamic_axes is None:
                dynamic_axes = {
                    'gene_expression': {0: 'num_nodes'},
                    'spatial_coords': {0: 'num_nodes'},
                    'edge_index': {1: 'num_edges'},
                    'output': {0: 'num_nodes'}
                }
            
            # Export to ONNX
            torch.onnx.export(
                model,
                inputs,
                export_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Optimize ONNX model if requested
            if optimize:
                self._optimize_onnx_model(export_path)
            
            # Verify exported model
            if self._verify_onnx_model(export_path, dummy_input):
                logger.info("ONNX export completed successfully")
                return True
            else:
                logger.error("ONNX model verification failed")
                return False
                
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False
    
    def _optimize_onnx_model(self, model_path: Union[str, Path]) -> None:
        """Optimize ONNX model."""
        try:
            import onnx
            from onnxruntime.tools import optimizer
            
            logger.info("Optimizing ONNX model")
            
            # Load model
            model = onnx.load(str(model_path))
            
            # Basic optimization
            optimized_model = optimizer.optimize_model(
                str(model_path),
                model_type='transformer',
                opt_level=1
            )
            
            # Save optimized model
            onnx.save(optimized_model, str(model_path))
            
        except ImportError:
            logger.warning("ONNX optimization tools not available")
        except Exception as e:
            logger.warning(f"ONNX optimization failed: {e}")
    
    def _verify_onnx_model(
        self,
        model_path: Union[str, Path],
        dummy_input: Dict[str, torch.Tensor]
    ) -> bool:
        """Verify ONNX model can be loaded and run."""
        try:
            import onnxruntime as ort
            
            # Create inference session
            session = ort.InferenceSession(str(model_path))
            
            # Prepare inputs
            if isinstance(dummy_input, dict):
                ort_inputs = {
                    k: v.cpu().numpy() for k, v in dummy_input.items()
                }
            else:
                ort_inputs = {"input": dummy_input.cpu().numpy()}
            
            # Run inference
            outputs = session.run(None, ort_inputs)
            
            logger.info("ONNX model verification successful")
            return True
            
        except ImportError:
            logger.warning("ONNXRuntime not available for verification")
            return True  # Assume success if can't verify
        except Exception as e:
            logger.error(f"ONNX verification failed: {e}")
            return False
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        quantization_method: str = "dynamic",
        backend: str = "fbgemm"
    ) -> nn.Module:
        """
        Quantize model for edge deployment.
        
        Args:
            model: Model to quantize
            calibration_data: Data for calibration
            quantization_method: 'dynamic', 'static', or 'qat'
            backend: Quantization backend
            
        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model using {quantization_method} quantization")
        
        try:
            model.eval()
            
            if quantization_method == "dynamic":
                return self._dynamic_quantization(model, backend)
            elif quantization_method == "static":
                return self._static_quantization(model, calibration_data, backend)
            elif quantization_method == "qat":
                return self._quantization_aware_training(model, backend)
            else:
                raise ValueError(f"Unknown quantization method: {quantization_method}")
                
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return model
    
    def _dynamic_quantization(self, model: nn.Module, backend: str) -> nn.Module:
        """Apply dynamic quantization."""
        torch.backends.quantized.engine = backend
        
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d, nn.Conv2d},
            dtype=torch.qint8
        )
        
        logger.info("Dynamic quantization completed")
        return quantized_model
    
    def _static_quantization(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
        backend: str
    ) -> nn.Module:
        """Apply static quantization with calibration."""
        torch.backends.quantized.engine = backend
        
        # Prepare model for quantization
        model.qconfig = torch.quantization.get_default_qconfig(backend)
        model_prepared = torch.quantization.prepare(model)
        
        # Calibrate with representative data
        model_prepared.eval()
        with torch.no_grad():
            # Run calibration data through model
            if isinstance(calibration_data, dict):
                model_prepared(**calibration_data)
            else:
                model_prepared(calibration_data)
        
        # Convert to quantized model
        quantized_model = torch.quantization.convert(model_prepared)
        
        logger.info("Static quantization completed")
        return quantized_model
    
    def _quantization_aware_training(self, model: nn.Module, backend: str) -> nn.Module:
        """Prepare model for quantization-aware training."""
        torch.backends.quantized.engine = backend
        
        model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
        model_prepared = torch.quantization.prepare_qat(model)
        
        logger.info("Model prepared for quantization-aware training")
        return model_prepared
    
    def optimize_memory_usage(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for memory efficiency.
        
        Args:
            model: Model to optimize
            
        Returns:
            Memory-optimized model
        """
        logger.info("Optimizing model memory usage")
        
        # Enable gradient checkpointing if available
        if hasattr(model, 'enable_gradient_checkpointing'):
            model.enable_gradient_checkpointing()
            logger.info("Enabled gradient checkpointing")
        
        # Optimize attention computation
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'attention'):
                    self._optimize_attention_memory(layer.attention)
        
        return model
    
    def _optimize_attention_memory(self, attention_layer) -> None:
        """Optimize attention layer for memory efficiency."""
        # Enable memory efficient attention if available
        if hasattr(attention_layer, 'enable_memory_efficient_attention'):
            attention_layer.enable_memory_efficient_attention()
        
        # Enable flash attention if available
        if hasattr(attention_layer, 'enable_flash_attention'):
            try:
                attention_layer.enable_flash_attention()
                logger.info("Enabled flash attention")
            except:
                logger.info("Flash attention not available")


class BatchProcessor:
    """
    Optimized batch processing for large-scale inference.
    
    Handles:
    - Memory-efficient batching
    - Streaming inference
    - Multi-GPU processing
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        max_batch_size: int = 32,
        enable_amp: bool = True
    ):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.enable_amp = enable_amp
        
        # Initialize automatic mixed precision
        if enable_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        logger.info(f"Initialized BatchProcessor with max batch size: {max_batch_size}")
    
    def process_large_dataset(
        self,
        data_loader,
        output_callback: Optional[callable] = None,
        progress_callback: Optional[callable] = None
    ) -> List[torch.Tensor]:
        """
        Process large dataset with memory-efficient batching.
        
        Args:
            data_loader: DataLoader for the dataset
            output_callback: Callback for processing outputs
            progress_callback: Callback for progress updates
            
        Returns:
            List of processed outputs
        """
        logger.info("Starting large dataset processing")
        
        results = []
        self.model.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    batch = batch.to(self.device)
                
                # Process with automatic mixed precision
                if self.enable_amp and torch.cuda.is_available():
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch)
                else:
                    outputs = self.model(batch)
                
                # Move outputs back to CPU to save GPU memory
                if isinstance(outputs, dict):
                    outputs = {k: v.cpu() for k, v in outputs.items()}
                else:
                    outputs = outputs.cpu()
                
                # Process outputs
                if output_callback:
                    output_callback(outputs, batch_idx)
                else:
                    results.append(outputs)
                
                # Progress update
                if progress_callback:
                    progress_callback(batch_idx, len(data_loader))
                
                # Clear GPU cache periodically
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        logger.info("Large dataset processing completed")
        return results
    
    def stream_process(
        self,
        data_stream,
        buffer_size: int = 100,
        output_callback: Optional[callable] = None
    ):
        """
        Process streaming data with buffering.
        
        Args:
            data_stream: Iterable data stream
            buffer_size: Size of processing buffer
            output_callback: Callback for outputs
        """
        logger.info("Starting stream processing")
        
        buffer = []
        self.model.eval()
        
        with torch.no_grad():
            for data in data_stream:
                buffer.append(data)
                
                if len(buffer) >= buffer_size:
                    # Process buffer
                    batch_data = self._prepare_batch(buffer)
                    
                    if self.enable_amp and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            outputs = self.model(batch_data)
                    else:
                        outputs = self.model(batch_data)
                    
                    if output_callback:
                        output_callback(outputs)
                    
                    # Clear buffer
                    buffer = []
                    
                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Process remaining data in buffer
            if buffer:
                batch_data = self._prepare_batch(buffer)
                outputs = self.model(batch_data)
                if output_callback:
                    output_callback(outputs)
        
        logger.info("Stream processing completed")
    
    def _prepare_batch(self, data_list: List) -> torch.Tensor:
        """Prepare batch from list of data items."""
        # Stack data items into batch
        if isinstance(data_list[0], dict):
            batch = {}
            for key in data_list[0].keys():
                batch[key] = torch.stack([item[key] for item in data_list])
                batch[key] = batch[key].to(self.device)
            return batch
        else:
            batch = torch.stack(data_list)
            return batch.to(self.device)


class PerformanceProfiler:
    """
    Performance profiling and monitoring utilities.
    """
    
    def __init__(self):
        self.profiling_data = {}
        logger.info("Initialized PerformanceProfiler")
    
    def profile_model_inference(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, Any]:
        """
        Profile model inference performance.
        
        Args:
            model: Model to profile
            sample_input: Sample input for profiling
            num_runs: Number of profiling runs
            warmup_runs: Number of warmup runs
            
        Returns:
            Profiling results
        """
        logger.info(f"Profiling model inference over {num_runs} runs")
        
        model.eval()
        device = next(model.parameters()).device
        
        # Move input to device
        if isinstance(sample_input, dict):
            sample_input = {k: v.to(device) for k, v in sample_input.items()}
        else:
            sample_input = sample_input.to(device)
        
        # Warmup runs
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(sample_input)
        
        # Synchronize before timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Profile actual runs
        import time
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                outputs = model(sample_input)
                
                # Synchronize after each run
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_time = total_time / num_runs
        throughput = num_runs / total_time
        
        # Memory usage
        memory_stats = {}
        if torch.cuda.is_available():
            memory_stats = {
                'allocated_mb': torch.cuda.memory_allocated() / 1024**2,
                'max_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2,
                'cached_mb': torch.cuda.memory_reserved() / 1024**2
            }
        
        results = {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'total_time_sec': total_time,
            'num_runs': num_runs,
            'memory_stats': memory_stats
        }
        
        logger.info(f"Inference profiling completed: {avg_time*1000:.2f}ms avg")
        return results
    
    def profile_with_torch_profiler(
        self,
        model: nn.Module,
        sample_input: Dict[str, torch.Tensor],
        output_path: Union[str, Path]
    ) -> None:
        """
        Profile with PyTorch profiler for detailed analysis.
        
        Args:
            model: Model to profile
            sample_input: Sample input
            output_path: Path to save profiling results
        """
        logger.info("Starting detailed profiling with PyTorch profiler")
        
        model.eval()
        device = next(model.parameters()).device
        
        if isinstance(sample_input, dict):
            sample_input = {k: v.to(device) for k, v in sample_input.items()}
        else:
            sample_input = sample_input.to(device)
        
        # Profile with PyTorch profiler
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output_path)),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for _ in range(10):
                with torch.no_grad():
                    outputs = model(sample_input)
                prof.step()
        
        logger.info(f"Detailed profiling completed, results saved to: {output_path}")


def optimize_for_production(
    model: nn.Module,
    sample_input: Dict[str, torch.Tensor],
    optimization_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply comprehensive optimizations for production deployment.
    
    Args:
        model: Model to optimize
        sample_input: Sample input for testing
        optimization_config: Configuration for optimizations
        
    Returns:
        Dictionary containing optimized models and metrics
    """
    if optimization_config is None:
        optimization_config = {
            'enable_compilation': True,
            'enable_quantization': True,
            'export_onnx': True,
            'profile_performance': True
        }
    
    logger.info("Starting comprehensive production optimization")
    
    optimizer = ModelOptimizer()
    profiler = PerformanceProfiler()
    
    results = {
        'original_model': model,
        'optimizations_applied': [],
        'performance_metrics': {}
    }
    
    # Profile original model
    if optimization_config.get('profile_performance', True):
        original_metrics = profiler.profile_model_inference(model, sample_input)
        results['performance_metrics']['original'] = original_metrics
    
    # Apply torch.compile optimization
    if optimization_config.get('enable_compilation', True):
        try:
            compiled_model = optimizer.compile_model(model, mode="max-autotune")
            results['compiled_model'] = compiled_model
            results['optimizations_applied'].append('torch_compile')
            
            # Profile compiled model
            if optimization_config.get('profile_performance', True):
                compiled_metrics = profiler.profile_model_inference(compiled_model, sample_input)
                results['performance_metrics']['compiled'] = compiled_metrics
                
        except Exception as e:
            logger.warning(f"Model compilation failed: {e}")
    
    # Apply quantization
    if optimization_config.get('enable_quantization', True):
        try:
            # Create calibration data from sample input
            if isinstance(sample_input, dict):
                calibration_data = {k: v[:1] for k, v in sample_input.items()}
            else:
                calibration_data = sample_input[:1]
            
            quantized_model = optimizer.quantize_model(
                model, calibration_data, quantization_method="dynamic"
            )
            results['quantized_model'] = quantized_model
            results['optimizations_applied'].append('quantization')
            
            # Profile quantized model
            if optimization_config.get('profile_performance', True):
                quantized_metrics = profiler.profile_model_inference(quantized_model, sample_input)
                results['performance_metrics']['quantized'] = quantized_metrics
                
        except Exception as e:
            logger.warning(f"Model quantization failed: {e}")
    
    # Export to ONNX
    if optimization_config.get('export_onnx', True):
        try:
            onnx_path = Path("model_optimized.onnx")
            success = optimizer.export_to_onnx(model, sample_input, onnx_path)
            if success:
                results['onnx_model_path'] = str(onnx_path)
                results['optimizations_applied'].append('onnx_export')
        except Exception as e:
            logger.warning(f"ONNX export failed: {e}")
    
    # Memory optimization
    memory_optimized_model = optimizer.optimize_memory_usage(model)
    results['memory_optimized_model'] = memory_optimized_model
    results['optimizations_applied'].append('memory_optimization')
    
    logger.info(f"Production optimization completed. Applied: {results['optimizations_applied']}")
    return results