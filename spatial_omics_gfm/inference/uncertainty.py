"""
Uncertainty quantification for spatial transcriptomics predictions.
Implements various uncertainty estimation methods including Monte Carlo dropout,
ensemble methods, and Bayesian approaches.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score
from anndata import AnnData

from .efficient_inference import EfficientInference
from ..models.graph_transformer import SpatialGraphTransformer

logger = logging.getLogger(__name__)


class UncertaintyQuantification:
    """
    Uncertainty quantification for spatial transcriptomics predictions.
    
    Implements multiple uncertainty estimation methods:
    - Monte Carlo Dropout
    - Deep Ensembles
    - Bayesian Neural Networks
    - Calibration techniques
    - Spatial uncertainty propagation
    """
    
    def __init__(
        self,
        model: SpatialGraphTransformer,
        uncertainty_method: str = "mc_dropout",
        num_samples: int = 100,
        ensemble_size: int = 5,
        dropout_rate: float = 0.1,
        device: Optional[str] = None,
        calibration_method: str = "temperature_scaling"
    ):
        """
        Initialize uncertainty quantification.
        
        Args:
            model: Pre-trained spatial graph transformer
            uncertainty_method: Method for uncertainty estimation
            num_samples: Number of samples for MC methods
            ensemble_size: Number of models in ensemble
            dropout_rate: Dropout rate for MC dropout
            device: Device for computations
            calibration_method: Method for probability calibration
        """
        self.model = model
        self.uncertainty_method = uncertainty_method
        self.num_samples = num_samples
        self.ensemble_size = ensemble_size
        self.dropout_rate = dropout_rate
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.calibration_method = calibration_method
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize components based on method
        if uncertainty_method == "ensemble":
            self.ensemble_models = self._create_ensemble()
        elif uncertainty_method == "bayesian":
            self._convert_to_bayesian()
        
        # Calibration
        self.calibrator = ProbabilityCalibrator(method=calibration_method)
        
        logger.info(f"Initialized UncertaintyQuantification with method: {uncertainty_method}")
    
    def predict_with_uncertainty(
        self,
        adata: AnnData,
        task_type: str = "cell_types",
        return_individual_predictions: bool = False,
        calibrate_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            adata: Spatial transcriptomics data
            task_type: Type of prediction task
            return_individual_predictions: Whether to return individual samples
            calibrate_probabilities: Whether to calibrate probabilities
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        logger.info(f"Predicting with uncertainty for task: {task_type}")
        
        if self.uncertainty_method == "mc_dropout":
            results = self._mc_dropout_prediction(adata, task_type)
        elif self.uncertainty_method == "ensemble":
            results = self._ensemble_prediction(adata, task_type)
        elif self.uncertainty_method == "bayesian":
            results = self._bayesian_prediction(adata, task_type)
        else:
            raise ValueError(f"Unknown uncertainty method: {self.uncertainty_method}")
        
        # Calibrate probabilities if requested
        if calibrate_probabilities and 'probabilities' in results:
            results['calibrated_probabilities'] = self.calibrator.calibrate(
                results['probabilities']
            )
        
        # Compute uncertainty metrics
        uncertainty_metrics = self._compute_uncertainty_metrics(results)
        results.update(uncertainty_metrics)
        
        # Spatial uncertainty analysis
        if 'spatial' in adata.obsm:
            spatial_uncertainty = self._analyze_spatial_uncertainty(
                adata, results['uncertainties']
            )
            results['spatial_uncertainty_analysis'] = spatial_uncertainty
        
        if not return_individual_predictions:
            # Remove individual predictions to save memory
            results.pop('individual_predictions', None)
        
        return results
    
    def _mc_dropout_prediction(self, adata: AnnData, task_type: str) -> Dict[str, Any]:
        """Monte Carlo Dropout uncertainty estimation."""
        logger.info("Performing Monte Carlo Dropout prediction")
        
        # Enable dropout during inference
        self._enable_mc_dropout()
        
        # Get input data
        gene_expression, spatial_coords, edge_index, edge_attr = self._prepare_inputs(adata)
        
        predictions = []
        embeddings_list = []
        
        # Multiple forward passes with dropout
        with torch.no_grad():
            for i in range(self.num_samples):
                # Forward pass with dropout
                model_output = self.model(
                    gene_expression=gene_expression,
                    spatial_coords=spatial_coords,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    return_embeddings=True
                )
                
                embeddings = model_output['embeddings']
                embeddings_list.append(embeddings.cpu().numpy())
                
                # Task-specific prediction
                if task_type == "cell_types" and hasattr(self.model, 'cell_type_head'):
                    logits = self.model.cell_type_head(embeddings)
                    probs = F.softmax(logits, dim=-1)
                    predictions.append(probs.cpu().numpy())
                elif task_type == "embeddings":
                    predictions.append(embeddings.cpu().numpy())
                else:
                    logger.warning(f"Task {task_type} not supported, using embeddings")
                    predictions.append(embeddings.cpu().numpy())
        
        # Disable dropout
        self._disable_mc_dropout()
        
        # Process predictions
        predictions = np.array(predictions)  # [num_samples, num_cells, num_classes/features]
        
        # Compute statistics
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        # Uncertainty measures
        if task_type == "cell_types":
            # Predictive entropy
            mean_probs = mean_prediction
            predictive_entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8), axis=1)
            
            # Mutual information (epistemic uncertainty)
            individual_entropies = -np.sum(predictions * np.log(predictions + 1e-8), axis=2)
            mean_entropy = np.mean(individual_entropies, axis=0)
            mutual_information = predictive_entropy - mean_entropy
            
            uncertainties = {
                'predictive_entropy': predictive_entropy,
                'mutual_information': mutual_information,
                'prediction_std': np.mean(std_prediction, axis=1)
            }
        else:
            # For continuous outputs (embeddings)
            uncertainties = {
                'prediction_std': np.mean(std_prediction, axis=1),
                'prediction_variance': np.mean(np.var(predictions, axis=0), axis=1)
            }
        
        return {
            'predictions': mean_prediction,
            'probabilities': mean_prediction if task_type == "cell_types" else None,
            'uncertainties': uncertainties,
            'prediction_std': std_prediction,
            'individual_predictions': predictions,
            'embeddings_samples': embeddings_list
        }
    
    def _ensemble_prediction(self, adata: AnnData, task_type: str) -> Dict[str, Any]:
        """Deep Ensemble uncertainty estimation."""
        logger.info("Performing Deep Ensemble prediction")
        
        # Get input data
        gene_expression, spatial_coords, edge_index, edge_attr = self._prepare_inputs(adata)
        
        predictions = []
        embeddings_list = []
        
        # Predictions from each ensemble member
        with torch.no_grad():
            for model in self.ensemble_models:
                model_output = model(
                    gene_expression=gene_expression,
                    spatial_coords=spatial_coords,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    return_embeddings=True
                )
                
                embeddings = model_output['embeddings']
                embeddings_list.append(embeddings.cpu().numpy())
                
                # Task-specific prediction
                if task_type == "cell_types" and hasattr(model, 'cell_type_head'):
                    logits = model.cell_type_head(embeddings)
                    probs = F.softmax(logits, dim=-1)
                    predictions.append(probs.cpu().numpy())
                elif task_type == "embeddings":
                    predictions.append(embeddings.cpu().numpy())
                else:
                    predictions.append(embeddings.cpu().numpy())
        
        # Process ensemble predictions
        predictions = np.array(predictions)  # [ensemble_size, num_cells, num_classes/features]
        
        # Compute statistics
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        # Ensemble uncertainties
        if task_type == "cell_types":
            # Agreement-based uncertainty
            predicted_classes = np.argmax(predictions, axis=2)
            agreement = np.array([
                np.sum(predicted_classes == predicted_classes[0], axis=0) / len(self.ensemble_models)
                for _ in range(len(self.ensemble_models))
            ])
            mean_agreement = np.mean(agreement, axis=0)
            
            uncertainties = {
                'ensemble_disagreement': 1 - mean_agreement,
                'prediction_std': np.mean(std_prediction, axis=1),
                'ensemble_variance': np.var(predictions, axis=0).mean(axis=1)
            }
        else:
            uncertainties = {
                'prediction_std': np.mean(std_prediction, axis=1),
                'ensemble_variance': np.var(predictions, axis=0).mean(axis=1)
            }
        
        return {
            'predictions': mean_prediction,
            'probabilities': mean_prediction if task_type == "cell_types" else None,
            'uncertainties': uncertainties,
            'prediction_std': std_prediction,
            'individual_predictions': predictions,
            'embeddings_samples': embeddings_list
        }
    
    def _bayesian_prediction(self, adata: AnnData, task_type: str) -> Dict[str, Any]:
        """Bayesian Neural Network uncertainty estimation."""
        logger.info("Performing Bayesian prediction")
        
        # Get input data
        gene_expression, spatial_coords, edge_index, edge_attr = self._prepare_inputs(adata)
        
        predictions = []
        embeddings_list = []
        
        # Sample from posterior
        with torch.no_grad():
            for i in range(self.num_samples):
                # Sample weights from posterior
                self._sample_bayesian_weights()
                
                # Forward pass
                model_output = self.model(
                    gene_expression=gene_expression,
                    spatial_coords=spatial_coords,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    return_embeddings=True
                )
                
                embeddings = model_output['embeddings']
                embeddings_list.append(embeddings.cpu().numpy())
                
                # Task-specific prediction
                if task_type == "cell_types" and hasattr(self.model, 'cell_type_head'):
                    logits = self.model.cell_type_head(embeddings)
                    probs = F.softmax(logits, dim=-1)
                    predictions.append(probs.cpu().numpy())
                else:
                    predictions.append(embeddings.cpu().numpy())
        
        # Process Bayesian predictions
        predictions = np.array(predictions)
        
        # Compute statistics
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)
        
        # Bayesian uncertainties
        uncertainties = {
            'epistemic_uncertainty': np.var(predictions, axis=0).mean(axis=1),
            'prediction_std': np.mean(std_prediction, axis=1)
        }
        
        return {
            'predictions': mean_prediction,
            'probabilities': mean_prediction if task_type == "cell_types" else None,
            'uncertainties': uncertainties,
            'prediction_std': std_prediction,
            'individual_predictions': predictions,
            'embeddings_samples': embeddings_list
        }
    
    def _prepare_inputs(self, adata: AnnData) -> Tuple[torch.Tensor, ...]:
        """Prepare inputs for model inference."""
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            expression = adata.X.toarray()
        else:
            expression = adata.X
        
        gene_expression = torch.tensor(expression, dtype=torch.float32).to(self.device)
        spatial_coords = torch.tensor(adata.obsm['spatial'], dtype=torch.float32).to(self.device)
        
        # Get spatial graph
        if 'spatial_graph' in adata.uns:
            edge_index = torch.tensor(adata.uns['spatial_graph']['edge_index'], dtype=torch.long).to(self.device)
            edge_attr = torch.tensor(adata.uns['spatial_graph']['edge_attr'], dtype=torch.float32).to(self.device)
        else:
            # Create dummy graph
            n_cells = len(expression)
            edge_index = torch.tensor([[0], [0]], dtype=torch.long).to(self.device)
            edge_attr = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32).to(self.device)
        
        return gene_expression, spatial_coords, edge_index, edge_attr
    
    def _enable_mc_dropout(self) -> None:
        """Enable dropout for Monte Carlo sampling."""
        def enable_dropout(module):
            if isinstance(module, nn.Dropout):
                module.train()
                module.p = self.dropout_rate
        
        self.model.apply(enable_dropout)
    
    def _disable_mc_dropout(self) -> None:
        """Disable dropout after Monte Carlo sampling."""
        self.model.eval()
    
    def _create_ensemble(self) -> List[SpatialGraphTransformer]:
        """Create ensemble of models."""
        logger.info(f"Creating ensemble of {self.ensemble_size} models")
        
        ensemble_models = []
        
        for i in range(self.ensemble_size):
            # Create model copy
            model_copy = type(self.model)(self.model.config)
            model_copy.load_state_dict(self.model.state_dict())
            
            # Add noise to weights for diversity
            with torch.no_grad():
                for param in model_copy.parameters():
                    param.add_(torch.randn_like(param) * 0.01)
            
            model_copy = model_copy.to(self.device)
            model_copy.eval()
            ensemble_models.append(model_copy)
        
        return ensemble_models
    
    def _convert_to_bayesian(self) -> None:
        """Convert model to Bayesian version."""
        logger.info("Converting model to Bayesian")
        
        # Simple approximation: add weight uncertainty
        self.weight_means = {}
        self.weight_stds = {}
        
        for name, param in self.model.named_parameters():
            self.weight_means[name] = param.data.clone()
            self.weight_stds[name] = torch.ones_like(param.data) * 0.01
    
    def _sample_bayesian_weights(self) -> None:
        """Sample weights from Bayesian posterior."""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.weight_means:
                    noise = torch.randn_like(param) * self.weight_stds[name]
                    param.data = self.weight_means[name] + noise
    
    def _compute_uncertainty_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute additional uncertainty metrics."""
        uncertainties = results['uncertainties']
        
        metrics = {}
        
        # Aggregate uncertainty measures
        if 'predictive_entropy' in uncertainties:
            metrics['mean_predictive_entropy'] = np.mean(uncertainties['predictive_entropy'])
            metrics['max_predictive_entropy'] = np.max(uncertainties['predictive_entropy'])
        
        if 'prediction_std' in uncertainties:
            metrics['mean_prediction_std'] = np.mean(uncertainties['prediction_std'])
            metrics['high_uncertainty_fraction'] = np.mean(
                uncertainties['prediction_std'] > np.percentile(uncertainties['prediction_std'], 75)
            )
        
        # Confidence intervals
        if 'individual_predictions' in results:
            predictions = results['individual_predictions']
            
            # Compute percentile-based confidence intervals
            ci_lower = np.percentile(predictions, 2.5, axis=0)
            ci_upper = np.percentile(predictions, 97.5, axis=0)
            ci_width = ci_upper - ci_lower
            
            metrics['confidence_intervals'] = {
                'lower': ci_lower,
                'upper': ci_upper,
                'width': ci_width,
                'mean_width': np.mean(ci_width)
            }
        
        return {'uncertainty_metrics': metrics}
    
    def _analyze_spatial_uncertainty(
        self,
        adata: AnnData,
        uncertainties: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze spatial patterns in uncertainty."""
        logger.info("Analyzing spatial uncertainty patterns")
        
        coords = adata.obsm['spatial']
        
        spatial_analysis = {}
        
        for uncertainty_type, uncertainty_values in uncertainties.items():
            if len(uncertainty_values) != len(coords):
                continue
            
            # Spatial autocorrelation of uncertainty
            spatial_autocorr = self._compute_spatial_autocorrelation(coords, uncertainty_values)
            
            # Uncertainty clusters
            uncertainty_clusters = self._find_uncertainty_clusters(coords, uncertainty_values)
            
            # Uncertainty gradients
            uncertainty_gradients = self._compute_uncertainty_gradients(coords, uncertainty_values)
            
            spatial_analysis[uncertainty_type] = {
                'spatial_autocorrelation': spatial_autocorr,
                'uncertainty_clusters': uncertainty_clusters,
                'uncertainty_gradients': uncertainty_gradients
            }
        
        return spatial_analysis
    
    def _compute_spatial_autocorrelation(self, coords: np.ndarray, values: np.ndarray) -> Dict[str, float]:
        """Compute spatial autocorrelation (Moran's I)."""
        from scipy.spatial.distance import pdist, squareform
        
        # Compute distance matrix
        distances = squareform(pdist(coords))
        
        # Create spatial weights (inverse distance)
        weights = 1.0 / (distances + 1e-8)
        np.fill_diagonal(weights, 0)
        
        # Normalize weights
        row_sums = np.sum(weights, axis=1)
        weights = weights / row_sums[:, np.newaxis]
        
        # Compute Moran's I
        n = len(values)
        mean_val = np.mean(values)
        
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val) ** 2
        
        morans_i = numerator / (denominator / n) if denominator > 0 else 0
        
        return {'morans_i': morans_i}
    
    def _find_uncertainty_clusters(self, coords: np.ndarray, uncertainties: np.ndarray) -> Dict[str, Any]:
        """Find clusters of high/low uncertainty."""
        from sklearn.cluster import DBSCAN
        
        # Identify high uncertainty points
        high_uncertainty_threshold = np.percentile(uncertainties, 75)
        high_uncertainty_mask = uncertainties > high_uncertainty_threshold
        
        if np.sum(high_uncertainty_mask) < 3:
            return {'num_clusters': 0, 'cluster_info': []}
        
        high_uncertainty_coords = coords[high_uncertainty_mask]
        
        # Cluster high uncertainty regions
        clustering = DBSCAN(eps=50, min_samples=3).fit(high_uncertainty_coords)
        
        cluster_info = []
        for cluster_id in np.unique(clustering.labels_):
            if cluster_id == -1:  # Noise
                continue
            
            cluster_mask = clustering.labels_ == cluster_id
            cluster_coords = high_uncertainty_coords[cluster_mask]
            
            cluster_info.append({
                'cluster_id': cluster_id,
                'num_points': np.sum(cluster_mask),
                'centroid': np.mean(cluster_coords, axis=0),
                'area': self._compute_cluster_area(cluster_coords)
            })
        
        return {
            'num_clusters': len(cluster_info),
            'cluster_info': cluster_info
        }
    
    def _compute_uncertainty_gradients(self, coords: np.ndarray, uncertainties: np.ndarray) -> Dict[str, Any]:
        """Compute spatial gradients of uncertainty."""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(coords)
        gradients = []
        
        for i, coord in enumerate(coords):
            # Find nearest neighbors
            distances, indices = tree.query(coord, k=7)  # Include self + 6 neighbors
            
            if len(indices) >= 3:
                neighbor_coords = coords[indices[1:]]  # Exclude self
                neighbor_uncertainties = uncertainties[indices[1:]]
                
                # Compute gradient using least squares
                A = np.column_stack([
                    neighbor_coords[:, 0] - coord[0],
                    neighbor_coords[:, 1] - coord[1],
                    np.ones(len(neighbor_coords))
                ])
                b = neighbor_uncertainties - uncertainties[i]
                
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    gradient_magnitude = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
                    gradients.append(gradient_magnitude)
                except:
                    gradients.append(0.0)
            else:
                gradients.append(0.0)
        
        gradients = np.array(gradients)
        
        return {
            'mean_gradient': np.mean(gradients),
            'max_gradient': np.max(gradients),
            'gradient_values': gradients
        }
    
    def _compute_cluster_area(self, coords: np.ndarray) -> float:
        """Compute area of coordinate cluster using convex hull."""
        if len(coords) < 3:
            return 0.0
        
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(coords)
            return hull.volume  # In 2D, volume is area
        except:
            return 0.0
    
    def calibrate_model(
        self,
        calibration_data: AnnData,
        task_type: str = "cell_types",
        true_labels: Optional[np.ndarray] = None
    ) -> None:
        """
        Calibrate uncertainty estimates on validation data.
        
        Args:
            calibration_data: Data for calibration
            task_type: Type of prediction task
            true_labels: True labels for calibration
        """
        logger.info("Calibrating uncertainty estimates")
        
        # Get predictions with uncertainty
        results = self.predict_with_uncertainty(
            calibration_data, task_type, calibrate_probabilities=False
        )
        
        if task_type == "cell_types" and true_labels is not None:
            # Calibrate probabilities
            self.calibrator.fit(results['probabilities'], true_labels)
            logger.info("Probability calibration completed")
    
    def evaluate_uncertainty_quality(
        self,
        test_data: AnnData,
        true_labels: np.ndarray,
        task_type: str = "cell_types"
    ) -> Dict[str, Any]:
        """
        Evaluate quality of uncertainty estimates.
        
        Args:
            test_data: Test data
            true_labels: True labels for evaluation
            task_type: Type of prediction task
            
        Returns:
            Uncertainty quality metrics
        """
        logger.info("Evaluating uncertainty quality")
        
        # Get predictions with uncertainty
        results = self.predict_with_uncertainty(test_data, task_type)
        
        predictions = results['predictions']
        uncertainties = results['uncertainties']
        
        metrics = {}
        
        if task_type == "cell_types":
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Accuracy vs uncertainty correlation
            correct_predictions = (predicted_classes == true_labels).astype(float)
            
            for uncertainty_type, uncertainty_values in uncertainties.items():
                # Correlation between uncertainty and correctness
                correlation = np.corrcoef(uncertainty_values, 1 - correct_predictions)[0, 1]
                metrics[f'{uncertainty_type}_correctness_correlation'] = correlation
                
                # Uncertainty calibration (reliability diagram)
                calibration_metrics = self._compute_calibration_metrics(
                    uncertainty_values, correct_predictions
                )
                metrics[f'{uncertainty_type}_calibration'] = calibration_metrics
        
        return metrics
    
    def _compute_calibration_metrics(
        self,
        uncertainties: np.ndarray,
        correctness: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """Compute calibration metrics for uncertainty estimates."""
        # Bin uncertainties
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(uncertainties, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for i in range(n_bins):
            bin_mask = bin_indices == i
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(correctness[bin_mask])
                bin_confidence = np.mean(uncertainties[bin_mask])
                bin_count = np.sum(bin_mask)
                
                bin_accuracies.append(bin_accuracy)
                bin_confidences.append(bin_confidence)
                bin_counts.append(bin_count)
        
        if not bin_accuracies:
            return {'ece': 0.0, 'mce': 0.0}
        
        bin_accuracies = np.array(bin_accuracies)
        bin_confidences = np.array(bin_confidences)
        bin_counts = np.array(bin_counts)
        
        # Expected Calibration Error (ECE)
        ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / np.sum(bin_counts)
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(bin_accuracies - bin_confidences))
        
        return {'ece': ece, 'mce': mce}


class ProbabilityCalibrator:
    """Calibrate probability predictions."""
    
    def __init__(self, method: str = "temperature_scaling"):
        self.method = method
        self.calibrator = None
        
        if method == "temperature_scaling":
            self.calibrator = TemperatureScaling()
        elif method == "platt_scaling":
            self.calibrator = PlattScaling()
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """Fit calibrator on validation data."""
        self.calibrator.fit(probabilities, true_labels)
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Calibrate probabilities."""
        if self.calibrator is None:
            logger.warning("Calibrator not fitted, returning original probabilities")
            return probabilities
        
        return self.calibrator.predict(probabilities)


class TemperatureScaling:
    """Temperature scaling calibration."""
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """Fit temperature parameter."""
        from scipy.optimize import minimize_scalar
        
        def nll_loss(temperature):
            # Convert to logits (approximately)
            logits = np.log(probabilities + 1e-8)
            scaled_logits = logits / temperature
            
            # Compute negative log likelihood
            scaled_probs = self._softmax(scaled_logits)
            nll = -np.mean(np.log(scaled_probs[np.arange(len(true_labels)), true_labels] + 1e-8))
            return nll
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        
        logger.info(f"Fitted temperature: {self.temperature:.3f}")
    
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply temperature scaling."""
        logits = np.log(probabilities + 1e-8)
        scaled_logits = logits / self.temperature
        return self._softmax(scaled_logits)
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class PlattScaling:
    """Platt scaling calibration."""
    
    def __init__(self):
        self.sigmoid_params = None
    
    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """Fit sigmoid parameters."""
        from sklearn.linear_model import LogisticRegression
        
        # Convert probabilities to scores (max probability)
        scores = np.max(probabilities, axis=1).reshape(-1, 1)
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(scores, true_labels)
        
        self.sigmoid_params = (lr.coef_[0][0], lr.intercept_[0])
        
        logger.info(f"Fitted Platt scaling parameters: {self.sigmoid_params}")
    
    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply Platt scaling."""
        if self.sigmoid_params is None:
            return probabilities
        
        a, b = self.sigmoid_params
        scores = np.max(probabilities, axis=1)
        
        # Apply sigmoid scaling
        calibrated_scores = 1.0 / (1.0 + np.exp(a * scores + b))
        
        # Redistribute probabilities proportionally
        calibrated_probs = probabilities * calibrated_scores.reshape(-1, 1)
        calibrated_probs = calibrated_probs / np.sum(calibrated_probs, axis=1, keepdims=True)
        
        return calibrated_probs