"""
Metrics and evaluation utilities for spatial transcriptomics analysis.
Implements comprehensive metrics for model evaluation and quality assessment.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, adjusted_rand_score,
    normalized_mutual_info_score, silhouette_score, adjusted_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors
from anndata import AnnData

logger = logging.getLogger(__name__)


class SpatialMetrics:
    """
    Comprehensive metrics for spatial transcriptomics analysis.
    
    Includes:
    - Classification metrics
    - Spatial coherence metrics
    - Clustering metrics
    - Biological plausibility metrics
    - Uncertainty metrics
    """
    
    def __init__(self):
        logger.info("Initialized SpatialMetrics")
    
    def compute_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            class_names: Names of classes
            
        Returns:
            Dictionary of classification metrics
        """
        logger.debug("Computing classification metrics")
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_micro'] = precision_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(y_true, y_pred, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        precision_per_class = precision_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, labels=unique_classes, zero_division=0)
        
        class_metrics = {}
        for i, class_id in enumerate(unique_classes):
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f"class_{class_id}"
            class_metrics[class_name] = {
                'precision': precision_per_class[i],
                'recall': recall_per_class[i],
                'f1': f1_per_class[i],
                'support': np.sum(y_true == class_id)
            }
        
        metrics['per_class'] = class_metrics
        
        # Probability-based metrics
        if y_prob is not None:
            try:
                # Multi-class ROC AUC
                if len(unique_classes) > 2:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovr')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, average='macro', multi_class='ovo')
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                
                # Average precision
                metrics['average_precision'] = average_precision_score(
                    y_true, y_prob, average='macro' if len(unique_classes) > 2 else None
                )
                
                # Top-k accuracy
                for k in [1, 3, 5]:
                    if k <= len(unique_classes):
                        top_k_pred = np.argsort(y_prob, axis=1)[:, -k:]
                        top_k_acc = np.mean([y_true[i] in top_k_pred[i] for i in range(len(y_true))])
                        metrics[f'top_{k}_accuracy'] = top_k_acc
                
            except Exception as e:
                logger.warning(f"Failed to compute probability-based metrics: {e}")
        
        # Confusion matrix statistics
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        metrics['confusion_matrix'] = cm
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = np.mean([cm[i, i] / np.sum(cm[i, :]) for i in range(len(unique_classes)) if np.sum(cm[i, :]) > 0])
        
        return metrics
    
    def compute_spatial_coherence(
        self,
        coords: np.ndarray,
        labels: np.ndarray,
        method: str = 'moran_i',
        k_neighbors: int = 6
    ) -> Dict[str, float]:
        """
        Compute spatial coherence of predictions.
        
        Args:
            coords: Spatial coordinates [n_cells, 2]
            labels: Cell labels or predictions
            method: Method for coherence calculation
            k_neighbors: Number of neighbors for spatial weights
            
        Returns:
            Spatial coherence metrics
        """
        logger.debug(f"Computing spatial coherence using {method}")
        
        metrics = {}
        
        if method == 'moran_i':
            metrics['morans_i'] = self._compute_morans_i(coords, labels, k_neighbors)
        elif method == 'geary_c':
            metrics['geary_c'] = self._compute_geary_c(coords, labels, k_neighbors)
        elif method == 'lee_l':
            metrics['lee_l'] = self._compute_lee_l(coords, labels, k_neighbors)
        elif method == 'all':
            metrics['morans_i'] = self._compute_morans_i(coords, labels, k_neighbors)
            metrics['geary_c'] = self._compute_geary_c(coords, labels, k_neighbors)
            metrics['lee_l'] = self._compute_lee_l(coords, labels, k_neighbors)
        
        # Spatial clustering coefficient
        metrics['spatial_clustering'] = self._compute_spatial_clustering(coords, labels, k_neighbors)
        
        # Local spatial autocorrelation
        local_autocorr = self._compute_local_autocorrelation(coords, labels, k_neighbors)
        metrics['mean_local_autocorr'] = np.mean(local_autocorr)
        metrics['std_local_autocorr'] = np.std(local_autocorr)
        
        return metrics
    
    def _compute_morans_i(self, coords: np.ndarray, values: np.ndarray, k: int) -> float:
        """Compute Moran's I spatial autocorrelation."""
        if len(np.unique(values)) < 2:
            return 0.0
        
        # Build spatial weights matrix
        weights = self._build_spatial_weights(coords, k)
        
        # Compute Moran's I
        n = len(values)
        mean_val = np.mean(values)
        
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val) ** 2
        
        if denominator == 0:
            return 0.0
        
        W = np.sum(weights)
        if W == 0:
            return 0.0
        
        morans_i = (n / W) * (numerator / denominator)
        return morans_i
    
    def _compute_geary_c(self, coords: np.ndarray, values: np.ndarray, k: int) -> float:
        """Compute Geary's C spatial autocorrelation."""
        if len(np.unique(values)) < 2:
            return 1.0
        
        weights = self._build_spatial_weights(coords, k)
        
        n = len(values)
        mean_val = np.mean(values)
        
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (values[i] - values[j]) ** 2
            denominator += (values[i] - mean_val) ** 2
        
        if denominator == 0:
            return 1.0
        
        W = np.sum(weights)
        if W == 0:
            return 1.0
        
        geary_c = ((n - 1) / (2 * W)) * (numerator / denominator)
        return geary_c
    
    def _compute_lee_l(self, coords: np.ndarray, values: np.ndarray, k: int) -> float:
        """Compute Lee's L spatial autocorrelation."""
        if len(np.unique(values)) < 2:
            return 0.0
        
        weights = self._build_spatial_weights(coords, k)
        
        n = len(values)
        mean_val = np.mean(values)
        var_val = np.var(values)
        
        if var_val == 0:
            return 0.0
        
        numerator = 0
        
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
        
        W = np.sum(weights)
        if W == 0:
            return 0.0
        
        lee_l = (n / W) * (numerator / (n * var_val))
        return lee_l
    
    def _build_spatial_weights(self, coords: np.ndarray, k: int) -> np.ndarray:
        """Build spatial weights matrix using k-nearest neighbors."""
        n = len(coords)
        weights = np.zeros((n, n))
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        # Build weights matrix
        for i in range(n):
            for j in range(1, len(indices[i])):  # Skip self (index 0)
                neighbor_idx = indices[i][j]
                # Use inverse distance weighting
                weight = 1.0 / (distances[i][j] + 1e-8)
                weights[i, neighbor_idx] = weight
                weights[neighbor_idx, i] = weight  # Symmetric
        
        # Row-normalize weights
        row_sums = np.sum(weights, axis=1)
        weights = weights / (row_sums[:, np.newaxis] + 1e-8)
        
        return weights
    
    def _compute_spatial_clustering(self, coords: np.ndarray, labels: np.ndarray, k: int) -> float:
        """Compute spatial clustering coefficient."""
        if len(np.unique(labels)) < 2:
            return 1.0
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        # Compute clustering coefficient
        clustering_scores = []
        
        for i in range(len(coords)):
            neighbors = indices[i][1:]  # Exclude self
            
            # Count neighbors with same label
            same_label_neighbors = np.sum(labels[neighbors] == labels[i])
            clustering_score = same_label_neighbors / len(neighbors)
            clustering_scores.append(clustering_score)
        
        return np.mean(clustering_scores)
    
    def _compute_local_autocorrelation(self, coords: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
        """Compute local spatial autocorrelation for each point."""
        n = len(coords)
        local_autocorr = np.zeros(n)
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        mean_val = np.mean(values)
        
        for i in range(n):
            neighbors = indices[i][1:]  # Exclude self
            neighbor_values = values[neighbors]
            
            # Compute local Moran's I
            if len(neighbors) > 0:
                local_autocorr[i] = (values[i] - mean_val) * np.mean(neighbor_values - mean_val)
        
        return local_autocorr
    
    def compute_clustering_metrics(
        self,
        embeddings: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        predicted_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute clustering quality metrics.
        
        Args:
            embeddings: Cell embeddings
            true_labels: True cluster labels (if available)
            predicted_labels: Predicted cluster labels
            
        Returns:
            Clustering metrics
        """
        logger.debug("Computing clustering metrics")
        
        metrics = {}
        
        # Internal metrics (don't require true labels)
        if predicted_labels is not None:
            # Silhouette score
            try:
                metrics['silhouette_score'] = silhouette_score(embeddings, predicted_labels)
            except Exception as e:
                logger.warning(f"Failed to compute silhouette score: {e}")
                metrics['silhouette_score'] = 0.0
            
            # Calinski-Harabasz score
            try:
                from sklearn.metrics import calinski_harabasz_score
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, predicted_labels)
            except Exception as e:
                logger.warning(f"Failed to compute Calinski-Harabasz score: {e}")
                metrics['calinski_harabasz_score'] = 0.0
            
            # Davies-Bouldin score
            try:
                from sklearn.metrics import davies_bouldin_score
                metrics['davies_bouldin_score'] = davies_bouldin_score(embeddings, predicted_labels)
            except Exception as e:
                logger.warning(f"Failed to compute Davies-Bouldin score: {e}")
                metrics['davies_bouldin_score'] = float('inf')
        
        # External metrics (require true labels)
        if true_labels is not None and predicted_labels is not None:
            # Adjusted Rand Index
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)
            
            # Normalized Mutual Information
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, predicted_labels)
            
            # Adjusted Mutual Information
            metrics['adjusted_mutual_info'] = adjusted_mutual_info_score(true_labels, predicted_labels)
            
            # Homogeneity, Completeness, V-measure
            from sklearn.metrics import homogeneity_completeness_v_measure
            homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(true_labels, predicted_labels)
            metrics['homogeneity'] = homogeneity
            metrics['completeness'] = completeness
            metrics['v_measure'] = v_measure
        
        return metrics
    
    def compute_interaction_metrics(
        self,
        predicted_interactions: np.ndarray,
        true_interactions: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute interaction prediction metrics.
        
        Args:
            predicted_interactions: Predicted interaction probabilities
            true_interactions: True interaction labels (if available)
            coords: Spatial coordinates for spatial validation
            
        Returns:
            Interaction metrics
        """
        logger.debug("Computing interaction metrics")
        
        metrics = {}
        
        # Basic statistics
        metrics['mean_interaction_prob'] = np.mean(predicted_interactions)
        metrics['std_interaction_prob'] = np.std(predicted_interactions)
        metrics['max_interaction_prob'] = np.max(predicted_interactions)
        
        # Distribution metrics
        metrics['num_high_confidence'] = np.sum(predicted_interactions > 0.8)
        metrics['num_medium_confidence'] = np.sum((predicted_interactions > 0.5) & (predicted_interactions <= 0.8))
        metrics['num_low_confidence'] = np.sum(predicted_interactions <= 0.5)
        
        # Threshold-based metrics
        if true_interactions is not None:
            for threshold in [0.5, 0.7, 0.9]:
                pred_binary = (predicted_interactions > threshold).astype(int)
                
                metrics[f'accuracy_at_{threshold}'] = accuracy_score(true_interactions, pred_binary)
                metrics[f'precision_at_{threshold}'] = precision_score(true_interactions, pred_binary, zero_division=0)
                metrics[f'recall_at_{threshold}'] = recall_score(true_interactions, pred_binary, zero_division=0)
                metrics[f'f1_at_{threshold}'] = f1_score(true_interactions, pred_binary, zero_division=0)
            
            # AUC metrics
            try:
                metrics['roc_auc'] = roc_auc_score(true_interactions, predicted_interactions)
                metrics['average_precision'] = average_precision_score(true_interactions, predicted_interactions)
            except Exception as e:
                logger.warning(f"Failed to compute AUC metrics: {e}")
        
        # Spatial validation
        if coords is not None:
            metrics['spatial_interaction_coherence'] = self._compute_interaction_spatial_coherence(
                predicted_interactions, coords
            )
        
        return metrics
    
    def _compute_interaction_spatial_coherence(self, interactions: np.ndarray, coords: np.ndarray) -> float:
        """Compute spatial coherence of interaction predictions."""
        # Find spatial neighbors
        nbrs = NearestNeighbors(n_neighbors=7).fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        coherence_scores = []
        
        for i in range(len(coords)):
            neighbors = indices[i][1:]  # Exclude self
            neighbor_interactions = interactions[neighbors]
            
            # Compute correlation with neighbor interactions
            if len(neighbor_interactions) > 1:
                correlation = np.corrcoef(interactions[i], np.mean(neighbor_interactions))[0, 1]
                if not np.isnan(correlation):
                    coherence_scores.append(correlation)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def compute_pathway_metrics(
        self,
        pathway_scores: np.ndarray,
        true_pathway_activity: Optional[np.ndarray] = None,
        coords: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute pathway analysis metrics.
        
        Args:
            pathway_scores: Predicted pathway activity scores
            true_pathway_activity: True pathway activities (if available)
            coords: Spatial coordinates
            
        Returns:
            Pathway metrics
        """
        logger.debug("Computing pathway metrics")
        
        metrics = {}
        
        # Basic statistics
        metrics['mean_pathway_score'] = np.mean(pathway_scores)
        metrics['std_pathway_score'] = np.std(pathway_scores)
        metrics['pathway_score_range'] = np.ptp(pathway_scores)
        
        # Activity distribution
        metrics['num_active_pathways'] = np.sum(pathway_scores > 0.5)
        metrics['fraction_active_pathways'] = metrics['num_active_pathways'] / len(pathway_scores)
        
        # Correlation with true activities
        if true_pathway_activity is not None:
            metrics['pathway_correlation'] = np.corrcoef(pathway_scores, true_pathway_activity)[0, 1]
            metrics['pathway_mse'] = np.mean((pathway_scores - true_pathway_activity) ** 2)
            metrics['pathway_mae'] = np.mean(np.abs(pathway_scores - true_pathway_activity))
        
        # Spatial coherence of pathway activities
        if coords is not None and pathway_scores.ndim == 2:  # Multiple pathways
            spatial_coherences = []
            for pathway_idx in range(pathway_scores.shape[1]):
                coherence = self.compute_spatial_coherence(
                    coords, pathway_scores[:, pathway_idx], method='moran_i'
                )['morans_i']
                spatial_coherences.append(coherence)
            
            metrics['mean_pathway_spatial_coherence'] = np.mean(spatial_coherences)
            metrics['std_pathway_spatial_coherence'] = np.std(spatial_coherences)
        
        return metrics
    
    def compute_uncertainty_metrics(
        self,
        predictions: np.ndarray,
        uncertainties: np.ndarray,
        true_labels: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute uncertainty quantification metrics.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            true_labels: True labels (if available)
            
        Returns:
            Uncertainty metrics
        """
        logger.debug("Computing uncertainty metrics")
        
        metrics = {}
        
        # Basic uncertainty statistics
        metrics['mean_uncertainty'] = np.mean(uncertainties)
        metrics['std_uncertainty'] = np.std(uncertainties)
        metrics['max_uncertainty'] = np.max(uncertainties)
        metrics['min_uncertainty'] = np.min(uncertainties)
        
        # Uncertainty distribution
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            metrics[f'uncertainty_p{p}'] = np.percentile(uncertainties, p)
        
        # Correlation between uncertainty and correctness
        if true_labels is not None:
            if predictions.ndim > 1:  # Multi-class predictions
                predicted_classes = np.argmax(predictions, axis=1)
            else:
                predicted_classes = predictions
            
            correctness = (predicted_classes == true_labels).astype(float)
            
            # Higher uncertainty should correlate with lower correctness
            uncertainty_correctness_corr = np.corrcoef(uncertainties, 1 - correctness)[0, 1]
            metrics['uncertainty_correctness_correlation'] = uncertainty_correctness_corr
            
            # Calibration metrics
            calibration_metrics = self._compute_uncertainty_calibration(uncertainties, correctness)
            metrics.update(calibration_metrics)
        
        return metrics
    
    def _compute_uncertainty_calibration(self, uncertainties: np.ndarray, correctness: np.ndarray) -> Dict[str, float]:
        """Compute uncertainty calibration metrics."""
        # Bin uncertainties
        n_bins = 10
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
    
    def compute_biological_plausibility(
        self,
        predictions: Dict[str, np.ndarray],
        known_biology: Dict[str, Any],
        coords: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute biological plausibility metrics.
        
        Args:
            predictions: Model predictions (cell types, interactions, etc.)
            known_biology: Known biological constraints and relationships
            coords: Spatial coordinates
            
        Returns:
            Biological plausibility metrics
        """
        logger.debug("Computing biological plausibility metrics")
        
        metrics = {}
        
        # Cell type co-location plausibility
        if 'cell_types' in predictions and coords is not None:
            colocation_score = self._compute_colocation_plausibility(
                predictions['cell_types'], coords, known_biology.get('colocation_rules', {})
            )
            metrics['cell_type_colocation_score'] = colocation_score
        
        # Interaction plausibility
        if 'interactions' in predictions:
            interaction_score = self._compute_interaction_plausibility(
                predictions['interactions'], known_biology.get('known_interactions', {})
            )
            metrics['interaction_plausibility_score'] = interaction_score
        
        # Pathway coherence
        if 'pathway_scores' in predictions:
            pathway_coherence = self._compute_pathway_coherence(
                predictions['pathway_scores'], known_biology.get('pathway_relationships', {})
            )
            metrics['pathway_coherence_score'] = pathway_coherence
        
        return metrics
    
    def _compute_colocation_plausibility(
        self,
        cell_types: np.ndarray,
        coords: np.ndarray,
        colocation_rules: Dict[str, Any]
    ) -> float:
        """Compute cell type co-location plausibility."""
        if not colocation_rules:
            return 1.0  # No rules to violate
        
        # Find spatial neighbors
        nbrs = NearestNeighbors(n_neighbors=7).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        violations = 0
        total_checks = 0
        
        for i, cell_type in enumerate(cell_types):
            neighbors = indices[i][1:]  # Exclude self
            neighbor_types = cell_types[neighbors]
            
            # Check co-location rules
            for neighbor_type in neighbor_types:
                total_checks += 1
                
                # Check if this co-location is forbidden
                forbidden_pairs = colocation_rules.get('forbidden', [])
                if [cell_type, neighbor_type] in forbidden_pairs or [neighbor_type, cell_type] in forbidden_pairs:
                    violations += 1
        
        if total_checks == 0:
            return 1.0
        
        plausibility = 1.0 - (violations / total_checks)
        return max(0.0, plausibility)
    
    def _compute_interaction_plausibility(
        self,
        predicted_interactions: np.ndarray,
        known_interactions: Dict[str, Any]
    ) -> float:
        """Compute interaction prediction plausibility."""
        if not known_interactions:
            return 1.0
        
        # Simple implementation - would need more sophisticated biology knowledge
        # For now, just check if predicted interactions are within reasonable bounds
        reasonable_interactions = np.sum((predicted_interactions > 0.01) & (predicted_interactions < 0.99))
        total_interactions = len(predicted_interactions)
        
        if total_interactions == 0:
            return 1.0
        
        return reasonable_interactions / total_interactions
    
    def _compute_pathway_coherence(
        self,
        pathway_scores: np.ndarray,
        pathway_relationships: Dict[str, Any]
    ) -> float:
        """Compute pathway activity coherence."""
        if not pathway_relationships or pathway_scores.ndim != 2:
            return 1.0
        
        # Check correlation between related pathways
        related_pairs = pathway_relationships.get('positive_correlations', [])
        
        if not related_pairs:
            return 1.0
        
        correlations = []
        
        for pathway1_idx, pathway2_idx in related_pairs:
            if pathway1_idx < pathway_scores.shape[1] and pathway2_idx < pathway_scores.shape[1]:
                corr = np.corrcoef(pathway_scores[:, pathway1_idx], pathway_scores[:, pathway2_idx])[0, 1]
                if not np.isnan(corr):
                    correlations.append(max(0, corr))  # Positive correlations only
        
        return np.mean(correlations) if correlations else 1.0


def evaluate_model_performance(
    model_outputs: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
    coords: Optional[np.ndarray] = None,
    known_biology: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Comprehensive model performance evaluation.
    
    Args:
        model_outputs: Dictionary of model predictions
        ground_truth: Dictionary of true labels/values
        coords: Spatial coordinates
        known_biology: Known biological constraints
        
    Returns:
        Comprehensive evaluation report
    """
    logger.info("Evaluating model performance")
    
    metrics = SpatialMetrics()
    evaluation_report = {}
    
    # Classification metrics
    if 'cell_types' in model_outputs and 'cell_types' in ground_truth:
        cell_type_metrics = metrics.compute_classification_metrics(
            ground_truth['cell_types'],
            model_outputs['cell_types'],
            model_outputs.get('cell_type_probabilities')
        )
        evaluation_report['cell_type_classification'] = cell_type_metrics
        
        # Spatial coherence of cell type predictions
        if coords is not None:
            spatial_coherence = metrics.compute_spatial_coherence(
                coords, model_outputs['cell_types'], method='all'
            )
            evaluation_report['cell_type_spatial_coherence'] = spatial_coherence
    
    # Interaction metrics
    if 'interactions' in model_outputs:
        interaction_metrics = metrics.compute_interaction_metrics(
            model_outputs['interactions'],
            ground_truth.get('interactions'),
            coords
        )
        evaluation_report['interaction_prediction'] = interaction_metrics
    
    # Pathway metrics
    if 'pathway_scores' in model_outputs:
        pathway_metrics = metrics.compute_pathway_metrics(
            model_outputs['pathway_scores'],
            ground_truth.get('pathway_scores'),
            coords
        )
        evaluation_report['pathway_analysis'] = pathway_metrics
    
    # Uncertainty metrics
    if 'uncertainties' in model_outputs:
        uncertainty_metrics = metrics.compute_uncertainty_metrics(
            model_outputs.get('predictions', model_outputs['cell_types']),
            model_outputs['uncertainties'],
            ground_truth.get('cell_types')
        )
        evaluation_report['uncertainty_quantification'] = uncertainty_metrics
    
    # Biological plausibility
    if known_biology is not None:
        plausibility_metrics = metrics.compute_biological_plausibility(
            model_outputs, known_biology, coords
        )
        evaluation_report['biological_plausibility'] = plausibility_metrics
    
    # Overall performance summary
    evaluation_report['summary'] = _compute_performance_summary(evaluation_report)
    
    logger.info("Model performance evaluation completed")
    return evaluation_report


def _compute_performance_summary(evaluation_report: Dict[str, Any]) -> Dict[str, float]:
    """Compute overall performance summary."""
    summary = {}
    
    # Classification performance
    if 'cell_type_classification' in evaluation_report:
        summary['overall_accuracy'] = evaluation_report['cell_type_classification']['accuracy']
        summary['overall_f1'] = evaluation_report['cell_type_classification']['f1_macro']
    
    # Spatial coherence
    if 'cell_type_spatial_coherence' in evaluation_report:
        summary['spatial_coherence'] = evaluation_report['cell_type_spatial_coherence'].get('morans_i', 0.0)
    
    # Biological plausibility
    if 'biological_plausibility' in evaluation_report:
        plausibility_scores = [v for v in evaluation_report['biological_plausibility'].values() if isinstance(v, (int, float))]
        summary['biological_plausibility'] = np.mean(plausibility_scores) if plausibility_scores else 1.0
    
    # Compute weighted overall score
    weights = {'overall_accuracy': 0.4, 'spatial_coherence': 0.3, 'biological_plausibility': 0.3}
    weighted_scores = []
    
    for metric, weight in weights.items():
        if metric in summary:
            weighted_scores.append(weight * summary[metric])
    
    if weighted_scores:
        summary['weighted_overall_score'] = sum(weighted_scores) / sum(weights[m] for m in summary.keys() if m in weights)
    
    return summary