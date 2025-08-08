"""
Tests for spatial-omics GFM task modules.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from anndata import AnnData

from spatial_omics_gfm.tasks.interaction_prediction import InteractionPredictor
from spatial_omics_gfm.tasks.pathway_analysis import PathwayAnalyzer
from spatial_omics_gfm.tasks.tissue_segmentation import TissueSegmenter, RegionClassifier
from spatial_omics_gfm.tasks.base import BaseTask, TaskConfig

from tests.conftest import assert_tensor_equal


class TestTaskConfig:
    """Test task configuration."""
    
    def test_default_config(self):
        """Test default task configuration."""
        config = TaskConfig()
        
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.dropout == 0.1
        assert config.device == "auto"
    
    def test_custom_config(self):
        """Test custom task configuration."""
        config = TaskConfig(
            batch_size=64,
            learning_rate=1e-3,
            dropout=0.2,
            max_epochs=50
        )
        
        assert config.batch_size == 64
        assert config.learning_rate == 1e-3
        assert config.dropout == 0.2
        assert config.max_epochs == 50


class TestBaseTask:
    """Test base task functionality."""
    
    def test_initialization(self, sample_model):
        """Test base task initialization."""
        config = TaskConfig()
        task = BaseTask(sample_model, config)
        
        assert task.model == sample_model
        assert task.config == config
        assert hasattr(task, 'device')
    
    def test_get_embeddings(self, sample_model, sample_adata, device):
        """Test getting embeddings from model."""
        task = BaseTask(sample_model, TaskConfig())
        
        embeddings = task._get_embeddings(sample_adata)
        
        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape[0] == sample_adata.n_obs
        assert embeddings.shape[1] == sample_model.config.hidden_dim
    
    def test_prepare_data(self, sample_model, sample_adata):
        """Test data preparation."""
        task = BaseTask(sample_model, TaskConfig())
        
        data_dict = task._prepare_data(sample_adata)
        
        assert 'gene_expression' in data_dict
        assert 'spatial_coords' in data_dict
        assert 'edge_index' in data_dict
        assert 'edge_attr' in data_dict


class TestInteractionPredictor:
    """Test cell-cell interaction prediction."""
    
    def test_initialization(self, sample_model):
        """Test interaction predictor initialization."""
        config = TaskConfig()
        predictor = InteractionPredictor(
            sample_model, config,
            database='custom',
            interaction_radius=50.0
        )
        
        assert predictor.database == 'custom'
        assert predictor.interaction_radius == 50.0
        assert hasattr(predictor, 'interaction_head')
    
    def test_predict_interactions(self, sample_model, sample_adata):
        """Test interaction prediction."""
        predictor = InteractionPredictor(sample_model, TaskConfig())
        
        # Add cell type information
        sample_adata.obs['cell_type'] = np.random.choice(
            ['TypeA', 'TypeB', 'TypeC'], sample_adata.n_obs
        )
        
        results = predictor.predict(
            sample_adata,
            return_probabilities=True,
            spatial_analysis=True
        )
        
        assert 'interaction_probabilities' in results
        assert 'interaction_pairs' in results
        assert 'spatial_networks' in results
        
        # Check shapes
        probs = results['interaction_probabilities']
        assert probs.shape[0] == sample_adata.n_obs
    
    def test_analyze_ligand_receptor_pairs(self, sample_model, sample_adata):
        """Test ligand-receptor pair analysis."""
        predictor = InteractionPredictor(sample_model, TaskConfig())
        
        # Add cell type information
        cell_types = np.random.choice(['TypeA', 'TypeB'], sample_adata.n_obs)
        sample_adata.obs['cell_type'] = cell_types
        
        lr_results = predictor._analyze_ligand_receptor_pairs(sample_adata, cell_types)
        
        assert isinstance(lr_results, dict)
        assert 'lr_pairs' in lr_results
        assert 'interaction_scores' in lr_results
    
    def test_find_signaling_neighborhoods(self, sample_model, sample_adata):
        """Test signaling neighborhood identification."""
        predictor = InteractionPredictor(sample_model, TaskConfig())
        
        # Create mock interactions
        n_cells = sample_adata.n_obs
        interactions = np.random.beta(2, 5, n_cells)
        
        neighborhoods = predictor.find_signaling_neighborhoods(sample_adata, interactions)
        
        assert 'neighborhoods' in neighborhoods
        assert 'neighborhood_stats' in neighborhoods
    
    def test_compute_interaction_networks(self, sample_model, sample_adata):
        """Test interaction network computation."""
        predictor = InteractionPredictor(sample_model, TaskConfig())
        
        # Create mock interaction probabilities
        n_cells = sample_adata.n_obs
        interaction_probs = np.random.beta(2, 5, n_cells)
        
        networks = predictor._compute_interaction_networks(sample_adata, interaction_probs)
        
        assert 'adjacency_matrix' in networks
        assert 'network_stats' in networks
        
        # Check adjacency matrix shape
        adj_matrix = networks['adjacency_matrix']
        assert adj_matrix.shape == (n_cells, n_cells)


class TestPathwayAnalyzer:
    """Test pathway analysis functionality."""
    
    def test_initialization(self, sample_model):
        """Test pathway analyzer initialization."""
        analyzer = PathwayAnalyzer(
            sample_model, TaskConfig(),
            databases=['KEGG', 'Reactome'],
            pathway_threshold=0.05
        )
        
        assert analyzer.databases == ['KEGG', 'Reactome']
        assert analyzer.pathway_threshold == 0.05
        assert hasattr(analyzer, 'pathway_head')
    
    def test_predict_pathway_activity(self, sample_model, sample_adata):
        """Test pathway activity prediction."""
        analyzer = PathwayAnalyzer(sample_model, TaskConfig())
        
        results = analyzer.predict(
            sample_adata,
            spatial_analysis=True,
            return_gene_scores=True
        )
        
        assert 'pathway_scores' in results
        assert 'pathway_names' in results
        assert 'spatial_gradients' in results
        assert 'gene_contributions' in results
        
        # Check shapes
        pathway_scores = results['pathway_scores']
        assert pathway_scores.shape[0] == sample_adata.n_obs
    
    def test_analyze_pathway_gradients(self, sample_model, sample_adata):
        """Test pathway gradient analysis."""
        analyzer = PathwayAnalyzer(sample_model, TaskConfig())
        
        # Create mock pathway scores
        n_cells = sample_adata.n_obs
        n_pathways = 10
        pathway_scores = np.random.beta(3, 3, (n_cells, n_pathways))
        
        gradients = analyzer._analyze_pathway_gradients(sample_adata, pathway_scores)
        
        assert 'gradient_magnitudes' in gradients
        assert 'gradient_directions' in gradients
        assert gradients['gradient_magnitudes'].shape[0] == n_cells
    
    def test_find_coregulated_pathways(self, sample_model):
        """Test co-regulated pathway identification."""
        analyzer = PathwayAnalyzer(sample_model, TaskConfig())
        
        # Create mock pathway scores
        n_cells = 100
        n_pathways = 15
        pathway_scores = np.random.beta(3, 3, (n_cells, n_pathways))
        
        coregulated = analyzer._find_coregulated_pathways(pathway_scores)
        
        assert 'correlation_matrix' in coregulated
        assert 'pathway_clusters' in coregulated
        
        # Check correlation matrix shape
        corr_matrix = coregulated['correlation_matrix']
        assert corr_matrix.shape == (n_pathways, n_pathways)
    
    def test_compute_pathway_enrichment(self, sample_model, sample_adata):
        """Test pathway enrichment computation."""
        analyzer = PathwayAnalyzer(sample_model, TaskConfig())
        
        # Create mock gene scores
        n_genes = sample_adata.n_vars
        gene_scores = np.random.randn(n_genes)
        
        enrichment = analyzer._compute_pathway_enrichment(gene_scores, sample_adata.var_names)
        
        assert 'enrichment_scores' in enrichment
        assert 'significant_pathways' in enrichment


class TestTissueSegmenter:
    """Test tissue segmentation functionality."""
    
    def test_initialization(self, sample_model):
        """Test tissue segmenter initialization."""
        segmenter = TissueSegmenter(
            sample_model, TaskConfig(),
            segmentation_method='hierarchical',
            min_region_size=20,
            boundary_smoothing=True
        )
        
        assert segmenter.segmentation_method == 'hierarchical'
        assert segmenter.min_region_size == 20
        assert segmenter.boundary_smoothing == True
        assert hasattr(segmenter, 'segmentation_head')
    
    def test_predict_segmentation(self, sample_model, sample_adata):
        """Test tissue segmentation prediction."""
        segmenter = TissueSegmenter(sample_model, TaskConfig())
        
        results = segmenter.predict(
            sample_adata,
            num_regions=5,
            return_embeddings=False,
            return_boundaries=True
        )
        
        assert 'region_assignments' in results
        assert 'region_properties' in results
        assert 'region_boundaries' in results
        assert 'num_regions' in results
        
        # Check region assignments
        assignments = results['region_assignments']
        assert len(assignments) == sample_adata.n_obs
        assert len(np.unique(assignments)) <= 5
    
    def test_determine_optimal_regions(self, sample_model, sample_adata):
        """Test optimal region number determination."""
        segmenter = TissueSegmenter(sample_model, TaskConfig())
        
        # Get embeddings
        embeddings = segmenter._get_embeddings(sample_adata)
        coords = sample_adata.obsm['spatial']
        
        optimal_regions = segmenter._determine_optimal_regions(embeddings, coords)
        
        assert isinstance(optimal_regions, int)
        assert optimal_regions >= 2
        assert optimal_regions <= 20
    
    def test_hierarchical_segmentation(self, sample_model, sample_adata):
        """Test hierarchical clustering segmentation."""
        segmenter = TissueSegmenter(sample_model, TaskConfig())
        
        embeddings = segmenter._get_embeddings(sample_adata)
        
        assignments = segmenter._hierarchical_segmentation(sample_adata, embeddings, 4)
        
        assert len(assignments) == sample_adata.n_obs
        assert len(np.unique(assignments)) <= 4
    
    def test_spectral_segmentation(self, sample_model, sample_adata):
        """Test spectral clustering segmentation."""
        segmenter = TissueSegmenter(sample_model, TaskConfig())
        
        # Add spatial graph to adata
        from spatial_omics_gfm.data.graph_construction import SpatialGraphBuilder
        from spatial_omics_gfm.data.base import SpatialDataConfig
        
        graph_builder = SpatialGraphBuilder(SpatialDataConfig())
        edge_index, edge_attr, graph_info = graph_builder.build_spatial_graph(
            sample_adata.obsm['spatial'], method='knn', k=6
        )
        graph_builder.add_graph_to_adata(sample_adata, edge_index, edge_attr, graph_info)
        
        embeddings = segmenter._get_embeddings(sample_adata)
        
        assignments = segmenter._spectral_segmentation(sample_adata, embeddings, 3)
        
        assert len(assignments) == sample_adata.n_obs
        assert len(np.unique(assignments)) <= 3
    
    def test_compute_region_properties(self, sample_model, sample_adata):
        """Test region property computation."""
        segmenter = TissueSegmenter(sample_model, TaskConfig())
        
        # Create mock region assignments
        assignments = np.random.randint(0, 3, sample_adata.n_obs)
        
        properties = segmenter._compute_region_properties(sample_adata, assignments)
        
        assert isinstance(properties, dict)
        assert len(properties) <= 3  # Number of unique regions
        
        for region_id, props in properties.items():
            assert 'num_cells' in props
            assert 'centroid' in props
            assert 'area' in props
            assert 'mean_expression' in props
    
    def test_remove_small_regions(self, sample_model, sample_adata):
        """Test small region removal."""
        segmenter = TissueSegmenter(sample_model, TaskConfig(), min_region_size=50)
        
        # Create assignments with some small regions
        assignments = np.random.randint(0, 10, sample_adata.n_obs)
        
        filtered_assignments = segmenter._remove_small_regions(sample_adata, assignments)
        
        # Check that small regions were merged
        unique_regions, counts = np.unique(filtered_assignments, return_counts=True)
        assert np.all(counts >= segmenter.min_region_size)
    
    def test_compute_region_boundaries(self, sample_model, sample_adata):
        """Test region boundary computation."""
        segmenter = TissueSegmenter(sample_model, TaskConfig())
        
        # Add spatial graph
        from spatial_omics_gfm.data.graph_construction import SpatialGraphBuilder
        from spatial_omics_gfm.data.base import SpatialDataConfig
        
        graph_builder = SpatialGraphBuilder(SpatialDataConfig())
        edge_index, edge_attr, graph_info = graph_builder.build_spatial_graph(
            sample_adata.obsm['spatial'], method='knn', k=6
        )
        graph_builder.add_graph_to_adata(sample_adata, edge_index, edge_attr, graph_info)
        
        # Create mock assignments
        assignments = np.random.randint(0, 3, sample_adata.n_obs)
        
        boundaries = segmenter._compute_region_boundaries(sample_adata, assignments)
        
        assert isinstance(boundaries, dict)
        # Each boundary should be between two different regions
        for region_pair, boundary_cells in boundaries.items():
            assert len(region_pair) == 2
            assert region_pair[0] != region_pair[1]
            assert len(boundary_cells) > 0


class TestRegionClassifier:
    """Test region classification functionality."""
    
    def test_initialization(self, sample_model):
        """Test region classifier initialization."""
        classifier = RegionClassifier(sample_model, TaskConfig())
        
        assert hasattr(classifier, 'region_classifier')
        assert hasattr(classifier, 'segmentation_head')
    
    def test_classify_regions(self, sample_model, sample_adata):
        """Test region classification."""
        classifier = RegionClassifier(sample_model, TaskConfig())
        
        # Create mock region assignments
        assignments = np.random.randint(0, 3, sample_adata.n_obs)
        
        results = classifier.classify_regions(
            sample_adata,
            assignments,
            region_types=['cortex', 'white_matter', 'hippocampus']
        )
        
        assert 'region_classifications' in results
        assert 'region_features' in results
        assert 'region_types' in results
        
        # Check that each region got classified
        classifications = results['region_classifications']
        assert len(classifications) == len(np.unique(assignments))
    
    def test_compute_region_features(self, sample_model, sample_adata):
        """Test region feature computation."""
        classifier = RegionClassifier(sample_model, TaskConfig())
        
        embeddings = classifier._get_embeddings(sample_adata)
        assignments = np.random.randint(0, 2, sample_adata.n_obs)
        
        features = classifier._compute_region_features(sample_adata, assignments, embeddings)
        
        assert isinstance(features, dict)
        assert len(features) <= 2  # Number of unique regions
        
        for region_id, feature_dict in features.items():
            assert 'mean_embedding' in feature_dict
            assert 'spatial_spread' in feature_dict
            assert 'expression_diversity' in feature_dict
    
    def test_classify_single_region(self, sample_model):
        """Test single region classification."""
        classifier = RegionClassifier(sample_model, TaskConfig())
        
        # Create mock features
        features = {
            'expression_diversity': 500,
            'num_cells': 100,
            'spatial_spread': np.array([15.0, 20.0])
        }
        
        region_types = ['cortex', 'white_matter', 'hippocampus']
        
        classification = classifier._classify_single_region(features, region_types)
        
        assert classification in region_types


@pytest.mark.slow
class TestTasksIntegration:
    """Integration tests for task modules."""
    
    def test_interaction_to_pathway_pipeline(self, sample_model, sample_adata):
        """Test interaction prediction to pathway analysis pipeline."""
        # First predict interactions
        interaction_predictor = InteractionPredictor(sample_model, TaskConfig())
        
        # Add cell types
        sample_adata.obs['cell_type'] = np.random.choice(
            ['TypeA', 'TypeB'], sample_adata.n_obs
        )
        
        interaction_results = interaction_predictor.predict(sample_adata)
        
        # Then analyze pathways
        pathway_analyzer = PathwayAnalyzer(sample_model, TaskConfig())
        pathway_results = pathway_analyzer.predict(sample_adata)
        
        # Both should produce valid results
        assert 'interaction_probabilities' in interaction_results
        assert 'pathway_scores' in pathway_results
    
    def test_segmentation_to_classification_pipeline(self, sample_model, sample_adata):
        """Test segmentation to classification pipeline."""
        # First segment tissue
        segmenter = TissueSegmenter(sample_model, TaskConfig())
        segmentation_results = segmenter.predict(sample_adata, num_regions=3)
        
        # Then classify regions
        classifier = RegionClassifier(sample_model, TaskConfig())
        classification_results = classifier.classify_regions(
            sample_adata,
            segmentation_results['region_assignments']
        )
        
        # Check pipeline consistency
        assert len(classification_results['region_classifications']) == segmentation_results['num_regions']
    
    def test_complete_spatial_analysis_workflow(self, sample_model, sample_adata):
        """Test complete spatial analysis workflow."""
        # Add cell type information
        sample_adata.obs['cell_type'] = np.random.choice(
            ['Neuron', 'Astrocyte', 'Microglia'], sample_adata.n_obs
        )
        
        # 1. Tissue segmentation
        segmenter = TissueSegmenter(sample_model, TaskConfig())
        seg_results = segmenter.predict(sample_adata, num_regions=4)
        
        # 2. Interaction prediction
        interaction_predictor = InteractionPredictor(sample_model, TaskConfig())
        int_results = interaction_predictor.predict(sample_adata)
        
        # 3. Pathway analysis
        pathway_analyzer = PathwayAnalyzer(sample_model, TaskConfig())
        path_results = pathway_analyzer.predict(sample_adata)
        
        # 4. Region classification
        classifier = RegionClassifier(sample_model, TaskConfig())
        class_results = classifier.classify_regions(
            sample_adata, seg_results['region_assignments']
        )
        
        # Verify all components produced results
        assert 'region_assignments' in seg_results
        assert 'interaction_probabilities' in int_results
        assert 'pathway_scores' in path_results
        assert 'region_classifications' in class_results
        
        # All results should have consistent dimensions
        n_cells = sample_adata.n_obs
        assert len(seg_results['region_assignments']) == n_cells
        assert int_results['interaction_probabilities'].shape[0] == n_cells
        assert path_results['pathway_scores'].shape[0] == n_cells