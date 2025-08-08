"""
Data augmentation for spatial transcriptomics data.
Implements spatial and expression augmentations to improve model robustness.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import StandardScaler
from anndata import AnnData

logger = logging.getLogger(__name__)


class SpatialAugmentor:
    """
    Data augmentation for spatial transcriptomics datasets.
    
    Provides various augmentation techniques including:
    - Spatial transformations (rotation, translation, scaling)
    - Expression noise and dropout
    - Spatial jittering and elastic deformation
    - Random cropping and subsampling
    """
    
    def __init__(
        self,
        rotation_range: float = 45.0,
        translation_range: float = 0.1,
        scaling_range: Tuple[float, float] = (0.8, 1.2),
        expression_noise_std: float = 0.1,
        dropout_rate: float = 0.1,
        spatial_jitter_std: float = 5.0,
        preserve_topology: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize spatial augmentor.
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            translation_range: Maximum translation as fraction of extent
            scaling_range: Range of scaling factors (min, max)
            expression_noise_std: Standard deviation for expression noise
            dropout_rate: Rate for gene expression dropout
            spatial_jitter_std: Standard deviation for spatial jittering
            preserve_topology: Whether to preserve spatial topology
            random_seed: Random seed for reproducibility
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scaling_range = scaling_range
        self.expression_noise_std = expression_noise_std
        self.dropout_rate = dropout_rate
        self.spatial_jitter_std = spatial_jitter_std
        self.preserve_topology = preserve_topology
        
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        logger.info("Initialized SpatialAugmentor")
    
    def augment(
        self,
        adata: AnnData,
        augmentation_types: Optional[List[str]] = None,
        probability: float = 0.5
    ) -> AnnData:
        """
        Apply augmentations to spatial transcriptomics data.
        
        Args:
            adata: Input AnnData object
            augmentation_types: List of augmentation types to apply
            probability: Probability of applying each augmentation
            
        Returns:
            Augmented AnnData object
        """
        if augmentation_types is None:
            augmentation_types = [
                'rotation', 'translation', 'scaling', 
                'expression_noise', 'spatial_jitter'
            ]
        
        # Create copy to avoid modifying original
        adata_aug = adata.copy()
        
        # Apply spatial augmentations
        if 'rotation' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_rotation(adata_aug)
        
        if 'translation' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_translation(adata_aug)
        
        if 'scaling' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_scaling(adata_aug)
        
        if 'flip' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_flip(adata_aug)
        
        if 'spatial_jitter' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_spatial_jitter(adata_aug)
        
        # Apply expression augmentations
        if 'expression_noise' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_expression_noise(adata_aug)
        
        if 'dropout' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_dropout(adata_aug)
        
        if 'scaling_expression' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_expression_scaling(adata_aug)
        
        # Advanced augmentations
        if 'elastic_deformation' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_elastic_deformation(adata_aug)
        
        if 'random_crop' in augmentation_types and np.random.random() < probability:
            adata_aug = self.apply_random_crop(adata_aug)
        
        return adata_aug
    
    def apply_rotation(self, adata: AnnData, angle: Optional[float] = None) -> AnnData:
        """Apply random rotation to spatial coordinates."""
        if 'spatial' not in adata.obsm:
            logger.warning("No spatial coordinates found, skipping rotation")
            return adata
        
        if angle is None:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        coords = adata.obsm['spatial'].copy()
        
        # Center coordinates
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        
        # Apply rotation
        angle_rad = np.deg2rad(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        coords_rotated = coords_centered @ rotation_matrix.T
        coords_final = coords_rotated + center
        
        # Update coordinates
        adata.obsm['spatial'] = coords_final
        
        # Update spatial graph if it exists
        if 'spatial_graph' in adata.uns and not self.preserve_topology:
            self._update_spatial_graph(adata)
        
        logger.debug(f"Applied rotation: {angle:.2f} degrees")
        return adata
    
    def apply_translation(self, adata: AnnData, offset: Optional[np.ndarray] = None) -> AnnData:
        """Apply random translation to spatial coordinates."""
        if 'spatial' not in adata.obsm:
            logger.warning("No spatial coordinates found, skipping translation")
            return adata
        
        coords = adata.obsm['spatial'].copy()
        
        if offset is None:
            # Calculate translation range based on coordinate extent
            coord_range = np.ptp(coords, axis=0)
            max_offset = coord_range * self.translation_range
            offset = np.random.uniform(-max_offset, max_offset)
        
        # Apply translation
        coords_translated = coords + offset
        adata.obsm['spatial'] = coords_translated
        
        logger.debug(f"Applied translation: {offset}")
        return adata
    
    def apply_scaling(self, adata: AnnData, scale_factor: Optional[float] = None) -> AnnData:
        """Apply random scaling to spatial coordinates."""
        if 'spatial' not in adata.obsm:
            logger.warning("No spatial coordinates found, skipping scaling")
            return adata
        
        if scale_factor is None:
            scale_factor = np.random.uniform(*self.scaling_range)
        
        coords = adata.obsm['spatial'].copy()
        
        # Center coordinates
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        
        # Apply scaling
        coords_scaled = coords_centered * scale_factor
        coords_final = coords_scaled + center
        
        adata.obsm['spatial'] = coords_final
        
        # Update spatial graph if it exists
        if 'spatial_graph' in adata.uns and not self.preserve_topology:
            self._update_spatial_graph(adata)
        
        logger.debug(f"Applied scaling: {scale_factor:.3f}")
        return adata
    
    def apply_flip(self, adata: AnnData, axis: Optional[int] = None) -> AnnData:
        """Apply random flip to spatial coordinates."""
        if 'spatial' not in adata.obsm:
            logger.warning("No spatial coordinates found, skipping flip")
            return adata
        
        if axis is None:
            axis = np.random.choice([0, 1])  # Flip along x or y axis
        
        coords = adata.obsm['spatial'].copy()
        
        # Center coordinates
        center = np.mean(coords, axis=0)
        coords_centered = coords - center
        
        # Apply flip
        coords_centered[:, axis] *= -1
        coords_final = coords_centered + center
        
        adata.obsm['spatial'] = coords_final
        
        # Update spatial graph if it exists
        if 'spatial_graph' in adata.uns and not self.preserve_topology:
            self._update_spatial_graph(adata)
        
        logger.debug(f"Applied flip along axis {axis}")
        return adata
    
    def apply_spatial_jitter(self, adata: AnnData, std: Optional[float] = None) -> AnnData:
        """Apply spatial jittering to coordinates."""
        if 'spatial' not in adata.obsm:
            logger.warning("No spatial coordinates found, skipping spatial jitter")
            return adata
        
        if std is None:
            std = self.spatial_jitter_std
        
        coords = adata.obsm['spatial'].copy()
        
        # Add Gaussian noise
        noise = np.random.normal(0, std, coords.shape)
        coords_jittered = coords + noise
        
        adata.obsm['spatial'] = coords_jittered
        
        # Update spatial graph if it exists
        if 'spatial_graph' in adata.uns and not self.preserve_topology:
            self._update_spatial_graph(adata)
        
        logger.debug(f"Applied spatial jitter with std: {std}")
        return adata
    
    def apply_expression_noise(self, adata: AnnData, noise_std: Optional[float] = None) -> AnnData:
        """Add Gaussian noise to gene expression."""
        if noise_std is None:
            noise_std = self.expression_noise_std
        
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            expression = adata.X.toarray()
        else:
            expression = adata.X.copy()
        
        # Add noise proportional to expression level
        noise = np.random.normal(0, noise_std, expression.shape)
        expression_noisy = expression + noise * expression
        
        # Ensure non-negative values
        expression_noisy = np.maximum(expression_noisy, 0)
        
        # Update expression matrix
        adata.X = expression_noisy
        
        logger.debug(f"Applied expression noise with std: {noise_std}")
        return adata
    
    def apply_dropout(self, adata: AnnData, dropout_rate: Optional[float] = None) -> AnnData:
        """Apply dropout to gene expression."""
        if dropout_rate is None:
            dropout_rate = self.dropout_rate
        
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            expression = adata.X.toarray()
        else:
            expression = adata.X.copy()
        
        # Create dropout mask
        dropout_mask = np.random.random(expression.shape) < dropout_rate
        
        # Apply dropout
        expression_dropout = expression.copy()
        expression_dropout[dropout_mask] = 0
        
        # Update expression matrix
        adata.X = expression_dropout
        
        logger.debug(f"Applied dropout with rate: {dropout_rate}")
        return adata
    
    def apply_expression_scaling(self, adata: AnnData, scale_range: Tuple[float, float] = (0.8, 1.2)) -> AnnData:
        """Apply random scaling to expression levels."""
        # Get expression matrix
        if hasattr(adata.X, 'toarray'):
            expression = adata.X.toarray()
        else:
            expression = adata.X.copy()
        
        # Random scaling factor for each gene
        scale_factors = np.random.uniform(*scale_range, size=expression.shape[1])
        
        # Apply scaling
        expression_scaled = expression * scale_factors
        
        # Update expression matrix
        adata.X = expression_scaled
        
        logger.debug(f"Applied expression scaling with range: {scale_range}")
        return adata
    
    def apply_elastic_deformation(self, adata: AnnData, alpha: float = 10.0, sigma: float = 3.0) -> AnnData:
        """Apply elastic deformation to spatial coordinates."""
        if 'spatial' not in adata.obsm:
            logger.warning("No spatial coordinates found, skipping elastic deformation")
            return adata
        
        coords = adata.obsm['spatial'].copy()
        
        # Create displacement field
        shape = (50, 50)  # Grid resolution
        dx = ndimage.gaussian_filter(
            (np.random.random(shape) - 0.5) * alpha, sigma, mode="constant", cval=0
        )
        dy = ndimage.gaussian_filter(
            (np.random.random(shape) - 0.5) * alpha, sigma, mode="constant", cval=0
        )
        
        # Map coordinates to grid
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        x_grid = ((coords[:, 0] - x_min) / (x_max - x_min) * (shape[1] - 1)).astype(int)
        y_grid = ((coords[:, 1] - y_min) / (y_max - y_min) * (shape[0] - 1)).astype(int)
        
        # Clamp to valid indices
        x_grid = np.clip(x_grid, 0, shape[1] - 1)
        y_grid = np.clip(y_grid, 0, shape[0] - 1)
        
        # Apply deformation
        coords_deformed = coords.copy()
        coords_deformed[:, 0] += dx[y_grid, x_grid]
        coords_deformed[:, 1] += dy[y_grid, x_grid]
        
        adata.obsm['spatial'] = coords_deformed
        
        # Update spatial graph if it exists
        if 'spatial_graph' in adata.uns and not self.preserve_topology:
            self._update_spatial_graph(adata)
        
        logger.debug(f"Applied elastic deformation with alpha: {alpha}, sigma: {sigma}")
        return adata
    
    def apply_random_crop(self, adata: AnnData, crop_fraction: float = 0.8) -> AnnData:
        """Apply random cropping to the spatial region."""
        if 'spatial' not in adata.obsm:
            logger.warning("No spatial coordinates found, skipping random crop")
            return adata
        
        coords = adata.obsm['spatial']
        
        # Calculate bounding box
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        
        # Calculate crop dimensions
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        crop_x_size = x_range * crop_fraction
        crop_y_size = y_range * crop_fraction
        
        # Random crop position
        crop_x_start = x_min + np.random.uniform(0, x_range - crop_x_size)
        crop_y_start = y_min + np.random.uniform(0, y_range - crop_y_size)
        
        crop_x_end = crop_x_start + crop_x_size
        crop_y_end = crop_y_start + crop_y_size
        
        # Select cells within crop region
        in_crop = (
            (coords[:, 0] >= crop_x_start) & (coords[:, 0] <= crop_x_end) &
            (coords[:, 1] >= crop_y_start) & (coords[:, 1] <= crop_y_end)
        )
        
        if np.sum(in_crop) == 0:
            logger.warning("Crop region contains no cells, returning original data")
            return adata
        
        # Filter data
        adata_cropped = adata[in_crop].copy()
        
        logger.debug(f"Applied random crop: {np.sum(in_crop)}/{len(adata)} cells retained")
        return adata_cropped
    
    def _update_spatial_graph(self, adata: AnnData) -> None:
        """Update spatial graph after coordinate transformation."""
        try:
            from .graph_construction import SpatialGraphBuilder
            
            builder = SpatialGraphBuilder()
            coords = adata.obsm['spatial']
            
            # Rebuild spatial graph
            edge_index, edge_attr, graph_info = builder.build_spatial_graph(coords)
            builder.add_graph_to_adata(adata, edge_index, edge_attr, graph_info)
            
            logger.debug("Updated spatial graph after transformation")
            
        except Exception as e:
            logger.warning(f"Failed to update spatial graph: {e}")
    
    def augment_batch(
        self,
        adata_list: List[AnnData],
        augmentation_types: Optional[List[str]] = None,
        probability: float = 0.5,
        different_augmentations: bool = True
    ) -> List[AnnData]:
        """
        Apply augmentations to a batch of datasets.
        
        Args:
            adata_list: List of AnnData objects
            augmentation_types: List of augmentation types to apply
            probability: Probability of applying each augmentation
            different_augmentations: Whether to apply different augmentations to each dataset
            
        Returns:
            List of augmented AnnData objects
        """
        augmented_list = []
        
        for i, adata in enumerate(adata_list):
            if different_augmentations:
                # Use different random seed for each dataset
                current_seed = np.random.randint(0, 10000)
                np.random.seed(current_seed)
            
            augmented_adata = self.augment(adata, augmentation_types, probability)
            augmented_list.append(augmented_adata)
        
        return augmented_list
    
    def create_augmented_dataset(
        self,
        adata: AnnData,
        num_augmentations: int = 5,
        augmentation_types: Optional[List[str]] = None,
        probability: float = 0.7
    ) -> List[AnnData]:
        """
        Create multiple augmented versions of a dataset.
        
        Args:
            adata: Input AnnData object
            num_augmentations: Number of augmented versions to create
            augmentation_types: List of augmentation types to apply
            probability: Probability of applying each augmentation
            
        Returns:
            List containing original and augmented datasets
        """
        augmented_datasets = [adata.copy()]  # Include original
        
        for i in range(num_augmentations):
            # Use different random seed for each augmentation
            np.random.seed(i * 42)
            
            augmented_adata = self.augment(adata, augmentation_types, probability)
            augmented_datasets.append(augmented_adata)
        
        logger.info(f"Created {num_augmentations} augmented versions of dataset")
        return augmented_datasets
    
    def apply_test_time_augmentation(
        self,
        adata: AnnData,
        num_augmentations: int = 10,
        augmentation_types: Optional[List[str]] = None
    ) -> List[AnnData]:
        """
        Apply test-time augmentation for robust predictions.
        
        Args:
            adata: Input AnnData object
            num_augmentations: Number of augmented versions for TTA
            augmentation_types: List of augmentation types (spatial only for TTA)
            
        Returns:
            List of augmented datasets for ensemble prediction
        """
        if augmentation_types is None:
            # Only spatial augmentations for TTA
            augmentation_types = ['rotation', 'translation', 'scaling', 'flip']
        
        # Use lower probability for TTA
        probability = 0.3
        
        tta_datasets = []
        
        for i in range(num_augmentations):
            np.random.seed(i * 123)  # Deterministic for reproducibility
            
            augmented_adata = self.augment(adata, augmentation_types, probability)
            tta_datasets.append(augmented_adata)
        
        logger.info(f"Created {num_augmentations} TTA augmented versions")
        return tta_datasets