"""
Comprehensive Test Suite for Autonomous SDLC Components
Tests for progressive quality gates, adaptive learning, and hypothesis testing
"""
import asyncio
import json
import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

from spatial_omics_gfm.autonomous.sdlc_executor import (
    AutonomousSDLCExecutor,
    SDLCPhase,
    ExecutionStatus,
    PhaseResult
)
from spatial_omics_gfm.autonomous.adaptive_learning import (
    AdaptiveLearningSystem,
    LearningStrategy,
    AdaptationTrigger,
    LearningMetrics,
    AdaptationEvent
)
from spatial_omics_gfm.autonomous.hypothesis_testing import (
    HypothesisTestingFramework,
    Hypothesis,
    HypothesisType,
    ExperimentDesign,
    ExperimentResult,
    ExperimentStatus
)
from spatial_omics_gfm.quality.progressive_gates import (
    ProgressiveQualityGates,
    QualityMetric,
    GateStatus
)


class TestProgressiveQualityGates:
    """Test suite for Progressive Quality Gates"""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create basic project structure
            (project_path / "spatial_omics_gfm").mkdir()
            (project_path / "tests").mkdir()
            (project_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
            
            yield project_path
    
    @pytest.fixture
    def quality_gates(self, temp_project_dir):
        """Create ProgressiveQualityGates instance"""
        return ProgressiveQualityGates(temp_project_dir)
    
    def test_initialization(self, quality_gates):
        """Test quality gates initialization"""
        assert quality_gates is not None
        assert quality_gates.config is not None
        assert "code_quality" in quality_gates.config
        assert "security" in quality_gates.config
        assert "performance" in quality_gates.config
        assert "testing" in quality_gates.config
    
    def test_config_loading(self, temp_project_dir):
        """Test configuration loading"""
        custom_config = {
            "code_quality": {"min_coverage": 90.0},
            "security": {"enabled": False}
        }
        
        gates = ProgressiveQualityGates(temp_project_dir, custom_config)
        assert gates.config["code_quality"]["min_coverage"] == 90.0
        assert gates.config["security"]["enabled"] is False
    
    def test_code_quality_gate(self, quality_gates):
        """Test code quality gate execution"""
        with patch('subprocess.run') as mock_run:
            # Mock successful pytest with coverage
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            result = quality_gates._execute_code_quality_gate()
            
            assert result.gate_name == "code_quality"
            assert isinstance(result.status, GateStatus)
            assert len(result.metrics) > 0
            assert result.execution_time >= 0
    
    def test_security_gate(self, quality_gates):
        """Test security gate execution"""
        result = quality_gates._execute_security_gate()
        
        assert result.gate_name == "security"
        assert isinstance(result.status, GateStatus)
        assert len(result.metrics) > 0
        assert result.execution_time >= 0
    
    def test_performance_gate(self, quality_gates):
        """Test performance gate execution"""
        result = quality_gates._execute_performance_gate()
        
        assert result.gate_name == "performance"
        assert isinstance(result.status, GateStatus)
        assert len(result.metrics) > 0
        assert result.execution_time >= 0
    
    def test_testing_gate(self, quality_gates):
        """Test testing gate execution"""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
            
            result = quality_gates._execute_testing_gate()
            
            assert result.gate_name == "testing"
            assert isinstance(result.status, GateStatus)
            assert len(result.metrics) > 0
    
    def test_test_coverage_check(self, quality_gates):
        """Test test coverage checking"""
        with patch('subprocess.run') as mock_run:
            # Mock coverage.json file
            coverage_data = {
                "totals": {"percent_covered": 87.5}
            }
            
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(coverage_data)
                
                with patch.object(Path, 'exists', return_value=True):
                    mock_run.return_value = Mock(returncode=0)
                    
                    metric = quality_gates._check_test_coverage()
                    
                    assert metric.name == "test_coverage"
                    assert metric.actual_value == 87.5
                    assert metric.status == GateStatus.PASSED
    
    def test_linting_check(self, quality_gates):
        """Test code linting check"""
        with patch('subprocess.run') as mock_run:
            # Mock successful linting (no output)
            mock_run.return_value = Mock(returncode=0, stdout="")
            
            metric = quality_gates._check_linting()
            
            assert metric.name == "linting"
            assert metric.actual_value == 0
            assert metric.status == GateStatus.PASSED
    
    def test_memory_usage_check(self, quality_gates):
        """Test memory usage checking"""
        with patch('psutil.virtual_memory') as mock_memory:
            mock_memory.return_value = Mock(used=512 * 1024 * 1024)  # 512 MB
            
            metric = quality_gates._check_memory_usage()
            
            assert metric.name == "memory_usage"
            assert metric.actual_value == 512.0
            assert metric.status == GateStatus.PASSED
    
    def test_results_saving(self, quality_gates, temp_project_dir):
        """Test results saving to file"""
        # Create mock results
        result = Mock()
        result.gate_name = "test_gate"
        result.status = GateStatus.PASSED
        result.execution_time = 1.5
        result.timestamp = time.time()
        result.metrics = []
        
        quality_gates.results = [result]
        quality_gates._save_results()
        
        results_file = temp_project_dir / "quality_gate_results.json"
        assert results_file.exists()
        
        with open(results_file) as f:
            saved_data = json.load(f)
            assert "results" in saved_data
            assert len(saved_data["results"]) == 1
    
    @patch('subprocess.run')
    def test_execute_all_gates(self, mock_run, quality_gates):
        """Test executing all quality gates"""
        mock_run.return_value = Mock(returncode=0, stdout="", stderr="")
        
        result = quality_gates.execute_all_gates()
        
        assert isinstance(result, bool)
        assert len(quality_gates.results) > 0
    
    def test_quality_metric_creation(self):
        """Test QualityMetric creation and properties"""
        metric = QualityMetric(
            name="test_metric",
            description="Test metric description",
            threshold=0.8,
            actual_value=0.9,
            status=GateStatus.PASSED
        )
        
        assert metric.name == "test_metric"
        assert metric.description == "Test metric description"
        assert metric.threshold == 0.8
        assert metric.actual_value == 0.9
        assert metric.status == GateStatus.PASSED


class TestAutonomousSDLCExecutor:
    """Test suite for Autonomous SDLC Executor"""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sdlc_executor(self, temp_project_dir):
        """Create AutonomousSDLCExecutor instance"""
        return AutonomousSDLCExecutor(temp_project_dir)
    
    def test_initialization(self, sdlc_executor):
        """Test SDLC executor initialization"""
        assert sdlc_executor is not None
        assert sdlc_executor.config is not None
        assert sdlc_executor.quality_gates is not None
        assert sdlc_executor.phase_results == {}
        assert sdlc_executor.current_phase is None
    
    def test_config_loading(self, temp_project_dir):
        """Test configuration loading"""
        custom_config = {
            "autonomous_execution": False,
            "max_retries": 5
        }
        
        executor = AutonomousSDLCExecutor(temp_project_dir, custom_config)
        assert executor.config["autonomous_execution"] is False
        assert executor.config["max_retries"] == 5
    
    @pytest.mark.asyncio
    async def test_execute_analysis_phase(self, sdlc_executor):
        """Test analysis phase execution"""
        result = await sdlc_executor._execute_analysis()
        
        assert isinstance(result, PhaseResult)
        assert result.phase == SDLCPhase.ANALYSIS
        assert result.status == ExecutionStatus.COMPLETED
        assert result.execution_time >= 0
        assert len(result.artifacts) > 0
        assert result.metrics is not None
    
    @pytest.mark.asyncio
    async def test_execute_generation_1_phase(self, sdlc_executor):
        """Test Generation 1 phase execution"""
        result = await sdlc_executor._execute_generation_1()
        
        assert isinstance(result, PhaseResult)
        assert result.phase == SDLCPhase.GENERATION_1
        assert result.status == ExecutionStatus.COMPLETED
        assert result.execution_time >= 0
        assert "core_functionality.py" in result.artifacts
    
    @pytest.mark.asyncio
    async def test_execute_generation_2_phase(self, sdlc_executor):
        """Test Generation 2 phase execution"""
        result = await sdlc_executor._execute_generation_2()
        
        assert isinstance(result, PhaseResult)
        assert result.phase == SDLCPhase.GENERATION_2
        assert result.status == ExecutionStatus.COMPLETED
        assert result.quality_score >= 0.8  # Should be robust
    
    @pytest.mark.asyncio
    async def test_execute_generation_3_phase(self, sdlc_executor):
        """Test Generation 3 phase execution"""
        result = await sdlc_executor._execute_generation_3()
        
        assert isinstance(result, PhaseResult)
        assert result.phase == SDLCPhase.GENERATION_3
        assert result.status == ExecutionStatus.COMPLETED
        assert result.quality_score >= 0.85  # Should be optimized
    
    @pytest.mark.asyncio
    async def test_execute_testing_phase(self, sdlc_executor):
        """Test testing phase execution"""
        with patch.object(sdlc_executor.quality_gates, 'execute_all_gates', return_value=True):
            result = await sdlc_executor._execute_testing()
            
            assert isinstance(result, PhaseResult)
            assert result.phase == SDLCPhase.TESTING
            assert result.status == ExecutionStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_execute_deployment_phase(self, sdlc_executor):
        """Test deployment phase execution"""
        result = await sdlc_executor._execute_deployment()
        
        assert isinstance(result, PhaseResult)
        assert result.phase == SDLCPhase.DEPLOYMENT
        assert result.status == ExecutionStatus.COMPLETED
        assert "Dockerfile" in result.artifacts
    
    @pytest.mark.asyncio
    async def test_execute_monitoring_phase(self, sdlc_executor):
        """Test monitoring phase execution"""
        result = await sdlc_executor._execute_monitoring()
        
        assert isinstance(result, PhaseResult)
        assert result.phase == SDLCPhase.MONITORING
        assert result.status == ExecutionStatus.COMPLETED
        assert result.quality_score >= 0.9
    
    @pytest.mark.asyncio
    async def test_phase_retry_logic(self, sdlc_executor):
        """Test phase retry logic on failure"""
        # Mock a phase that fails initially then succeeds
        call_count = 0
        
        async def mock_phase():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return PhaseResult(
                    phase=SDLCPhase.ANALYSIS,
                    status=ExecutionStatus.FAILED,
                    execution_time=0.1,
                    quality_score=0.0,
                    artifacts=[],
                    metrics={}
                )
            else:
                return PhaseResult(
                    phase=SDLCPhase.ANALYSIS,
                    status=ExecutionStatus.COMPLETED,
                    execution_time=0.1,
                    quality_score=0.9,
                    artifacts=[],
                    metrics={}
                )
        
        with patch.object(sdlc_executor, '_execute_analysis', side_effect=mock_phase):
            success = await sdlc_executor._execute_phase_with_retry(SDLCPhase.ANALYSIS)
            
            assert success is True
            assert call_count == 2  # Should retry once
    
    def test_validate_phase_quality(self, sdlc_executor):
        """Test phase quality validation"""
        # Add a successful phase result
        phase_result = PhaseResult(
            phase=SDLCPhase.ANALYSIS,
            status=ExecutionStatus.COMPLETED,
            execution_time=1.0,
            quality_score=0.9,
            artifacts=[],
            metrics={}
        )
        
        sdlc_executor.phase_results[SDLCPhase.ANALYSIS] = phase_result
        
        # Test quality validation
        is_valid = sdlc_executor._validate_phase_quality(SDLCPhase.ANALYSIS)
        assert is_valid is True
        
        # Test with low quality score
        phase_result.quality_score = 0.5
        is_valid = sdlc_executor._validate_phase_quality(SDLCPhase.ANALYSIS)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_generate_execution_report(self, sdlc_executor, temp_project_dir):
        """Test execution report generation"""
        # Add mock phase results
        sdlc_executor.execution_start_time = time.time()
        sdlc_executor.phase_results[SDLCPhase.ANALYSIS] = PhaseResult(
            phase=SDLCPhase.ANALYSIS,
            status=ExecutionStatus.COMPLETED,
            execution_time=1.0,
            quality_score=0.9,
            artifacts=[],
            metrics={}
        )
        
        await sdlc_executor._generate_execution_report()
        
        report_file = temp_project_dir / "autonomous_sdlc_report.json"
        assert report_file.exists()
        
        with open(report_file) as f:
            report_data = json.load(f)
            assert "execution_summary" in report_data
            assert "phase_results" in report_data


class TestAdaptiveLearningSystem:
    """Test suite for Adaptive Learning System"""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing"""
        return nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 2)
        )
    
    @pytest.fixture
    def adaptive_learning(self, simple_model):
        """Create AdaptiveLearningSystem instance"""
        return AdaptiveLearningSystem(simple_model)
    
    @pytest.fixture
    def sample_dataloader(self):
        """Create sample DataLoader for testing"""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        inputs = torch.randn(100, 10)
        targets = torch.randint(0, 2, (100,))
        dataset = TensorDataset(inputs, targets)
        
        return DataLoader(dataset, batch_size=16, shuffle=True)
    
    def test_initialization(self, adaptive_learning):
        """Test adaptive learning system initialization"""
        assert adaptive_learning is not None
        assert adaptive_learning.model is not None
        assert adaptive_learning.config is not None
        assert len(adaptive_learning.learning_strategies) > 0
        assert adaptive_learning.performance_monitor is not None
    
    def test_config_loading(self, simple_model):
        """Test configuration loading"""
        custom_config = {
            "adaptation": {"min_improvement_threshold": 0.05},
            "online_learning": {"learning_rate": 1e-3}
        }
        
        system = AdaptiveLearningSystem(simple_model, custom_config)
        assert system.config["adaptation"]["min_improvement_threshold"] == 0.05
        assert system.config["online_learning"]["learning_rate"] == 1e-3
    
    def test_measure_performance(self, adaptive_learning):
        """Test performance measurement"""
        metrics = adaptive_learning._measure_current_performance()
        
        assert isinstance(metrics, LearningMetrics)
        assert 0 <= metrics.accuracy_improvement <= 1
        assert 0 <= metrics.convergence_speed <= 1
        assert 0 <= metrics.stability_score <= 1
    
    def test_online_learning_update(self, adaptive_learning, sample_dataloader):
        """Test online learning update"""
        result = adaptive_learning._online_learning_update(sample_dataloader)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "avg_loss" in result
        assert "batches_processed" in result
        assert result["batches_processed"] > 0
    
    def test_meta_learning_update(self, adaptive_learning, sample_dataloader):
        """Test meta-learning update"""
        result = adaptive_learning._meta_learning_update(sample_dataloader)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "meta_loss" in result
        assert "tasks_processed" in result
    
    def test_continual_learning_update(self, adaptive_learning, sample_dataloader):
        """Test continual learning update"""
        result = adaptive_learning._continual_learning_update(sample_dataloader)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "avg_loss" in result
        assert "replay_buffer_size" in result
    
    def test_self_supervised_update(self, adaptive_learning, sample_dataloader):
        """Test self-supervised learning update"""
        result = adaptive_learning._self_supervised_update(sample_dataloader)
        
        assert isinstance(result, dict)
        assert "success" in result
        assert "avg_loss" in result
        assert "batches_processed" in result
    
    def test_adaptation_event_creation(self, adaptive_learning, sample_dataloader):
        """Test adaptation event creation and logging"""
        event = adaptive_learning.adapt_to_new_data(
            sample_dataloader,
            LearningStrategy.ONLINE_LEARNING,
            AdaptationTrigger.NEW_DATA_AVAILABLE
        )
        
        assert isinstance(event, AdaptationEvent)
        assert event.strategy == LearningStrategy.ONLINE_LEARNING
        assert event.trigger == AdaptationTrigger.NEW_DATA_AVAILABLE
        assert event.adaptation_time >= 0
        assert isinstance(event.successful, bool)
    
    def test_performance_monitor(self, adaptive_learning):
        """Test performance monitoring"""
        monitor = adaptive_learning.performance_monitor
        
        # Update with some performance values
        for i in range(10):
            monitor.update(0.8 + 0.1 * np.random.random())
        
        # Check degradation detection
        degradation = monitor.check_degradation()
        assert isinstance(degradation, bool)
    
    def test_distribution_shift_detection(self, adaptive_learning, sample_dataloader):
        """Test distribution shift detection"""
        detector = adaptive_learning.distribution_detector
        
        # Test with sample data
        for batch_data in sample_dataloader:
            shift_detected = detector.detect_shift(batch_data)
            assert isinstance(shift_detected, bool)
            break  # Only test one batch
    
    def test_replay_buffer(self, adaptive_learning):
        """Test replay buffer functionality"""
        buffer = adaptive_learning.replay_buffer
        
        # Test storing data
        inputs = torch.randn(5, 10)
        targets = torch.randint(0, 2, (5,))
        buffer.store(inputs, targets)
        
        assert len(buffer) == 1  # One batch stored
        
        # Test sampling
        sampled_inputs, sampled_targets = buffer.sample(3)
        assert sampled_inputs.shape[0] <= 3  # Should return at most 3 samples
    
    def test_calculate_improvement(self, adaptive_learning):
        """Test improvement calculation"""
        before_metrics = LearningMetrics(
            accuracy_improvement=0.8,
            convergence_speed=0.7,
            stability_score=0.9
        )
        
        after_metrics = LearningMetrics(
            accuracy_improvement=0.85,
            convergence_speed=0.75,
            stability_score=0.95
        )
        
        improvement = adaptive_learning._calculate_improvement(before_metrics, after_metrics)
        assert isinstance(improvement, float)
        assert improvement > 0  # Should show improvement
    
    def test_adaptation_strategy_selection(self, adaptive_learning):
        """Test adaptation strategy selection"""
        strategies = [
            (AdaptationTrigger.PERFORMANCE_DEGRADATION, LearningStrategy.CONTINUAL_LEARNING),
            (AdaptationTrigger.DISTRIBUTION_SHIFT, LearningStrategy.META_LEARNING),
            (AdaptationTrigger.NEW_DATA_AVAILABLE, LearningStrategy.ONLINE_LEARNING),
            (AdaptationTrigger.USER_FEEDBACK, LearningStrategy.REINFORCEMENT),
            (AdaptationTrigger.SCHEDULED_UPDATE, LearningStrategy.SELF_SUPERVISED)
        ]
        
        for trigger, expected_strategy in strategies:
            selected_strategy = adaptive_learning._select_adaptation_strategy(trigger)
            assert selected_strategy == expected_strategy


class TestHypothesisTestingFramework:
    """Test suite for Hypothesis Testing Framework"""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def hypothesis_framework(self, temp_project_dir):
        """Create HypothesisTestingFramework instance"""
        return HypothesisTestingFramework(temp_project_dir)
    
    @pytest.fixture
    def sample_hypothesis(self):
        """Create sample hypothesis for testing"""
        return Hypothesis(
            hypothesis_id="test_hyp_001",
            title="Test Algorithm Performance",
            description="Testing if new algorithm improves performance",
            hypothesis_type=HypothesisType.ALGORITHMIC_IMPROVEMENT,
            null_hypothesis="No significant difference in performance",
            alternative_hypothesis="New algorithm performs significantly better",
            success_criteria={"accuracy": 0.05, "speed": 0.1},
            significance_level=0.05,
            expected_effect_size=0.3
        )
    
    def test_initialization(self, hypothesis_framework):
        """Test hypothesis testing framework initialization"""
        assert hypothesis_framework is not None
        assert hypothesis_framework.config is not None
        assert hypothesis_framework.hypotheses == {}
        assert hypothesis_framework.experiments == {}
        assert hypothesis_framework.results == {}
    
    def test_hypothesis_generation(self, hypothesis_framework):
        """Test hypothesis generation"""
        domain_knowledge = {
            "baselines": {"accuracy": 0.85, "speed": 100},
            "current_methods": ["method_a", "method_b"]
        }
        
        research_gaps = [
            "Improve model accuracy on spatial data",
            "Reduce computational complexity"
        ]
        
        hypotheses = hypothesis_framework.generate_research_hypotheses(
            domain_knowledge,
            research_gaps
        )
        
        assert len(hypotheses) > 0
        assert all(isinstance(h, Hypothesis) for h in hypotheses)
        assert len(hypothesis_framework.hypotheses) == len(hypotheses)
    
    def test_experiment_design(self, hypothesis_framework, sample_hypothesis):
        """Test experiment design"""
        # Add hypothesis to framework
        hypothesis_framework.hypotheses[sample_hypothesis.hypothesis_id] = sample_hypothesis
        
        experiment = hypothesis_framework.design_experiment(
            sample_hypothesis.hypothesis_id
        )
        
        assert isinstance(experiment, ExperimentDesign)
        assert experiment.hypothesis.hypothesis_id == sample_hypothesis.hypothesis_id
        assert len(experiment.dependent_variables) > 0
        assert len(experiment.independent_variables) > 0
        assert experiment.sample_size >= sample_hypothesis.minimum_sample_size
    
    def test_experiment_execution(self, hypothesis_framework, sample_hypothesis):
        """Test experiment execution"""
        # Add hypothesis and design experiment
        hypothesis_framework.hypotheses[sample_hypothesis.hypothesis_id] = sample_hypothesis
        experiment = hypothesis_framework.design_experiment(sample_hypothesis.hypothesis_id)
        hypothesis_framework.experiments[experiment.experiment_id] = experiment
        
        # Execute experiment
        result = hypothesis_framework.execute_experiment(experiment.experiment_id)
        
        assert isinstance(result, ExperimentResult)
        assert result.experiment_id == experiment.experiment_id
        assert result.hypothesis_id == sample_hypothesis.hypothesis_id
        assert result.status in [ExperimentStatus.COMPLETED, ExperimentStatus.FAILED]
        assert result.execution_time is not None
    
    def test_statistical_analysis(self, hypothesis_framework):
        """Test statistical analysis of results"""
        analyzer = hypothesis_framework.statistical_analyzer
        
        # Create mock baseline and proposed results
        baseline_results = {
            "accuracy": 0.80,
            "precision": 0.78,
            "recall": 0.82
        }
        
        proposed_results = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.87
        }
        
        # Create mock hypothesis
        hypothesis = Hypothesis(
            hypothesis_id="test_hyp",
            title="Test",
            description="Test",
            hypothesis_type=HypothesisType.ALGORITHMIC_IMPROVEMENT,
            null_hypothesis="No difference",
            alternative_hypothesis="Improvement exists",
            success_criteria={"accuracy": 0.05}
        )
        
        analysis = analyzer.analyze_results(baseline_results, proposed_results, hypothesis)
        
        assert isinstance(analysis, dict)
        assert "tests" in analysis
        assert "effect_size" in analysis
        assert "p_value" in analysis
        assert "confidence_interval" in analysis
    
    def test_comparative_study(self, hypothesis_framework):
        """Test comparative study execution"""
        baseline_methods = ["baseline_a", "baseline_b"]
        proposed_method = "proposed_method"
        datasets = ["dataset_1", "dataset_2"]
        metrics = ["accuracy", "f1_score"]
        
        study_results = hypothesis_framework.run_comparative_study(
            baseline_methods,
            proposed_method,
            datasets,
            metrics
        )
        
        assert isinstance(study_results, dict)
        assert "study_id" in study_results
        assert "results" in study_results
        assert "statistical_comparison" in study_results
        assert "summary" in study_results
        
        # Check that all methods were evaluated
        assert proposed_method in study_results["results"]
        for baseline in baseline_methods:
            assert baseline in study_results["results"]
    
    def test_reproducibility_validation(self, hypothesis_framework, sample_hypothesis):
        """Test reproducibility validation"""
        # Setup experiment
        hypothesis_framework.hypotheses[sample_hypothesis.hypothesis_id] = sample_hypothesis
        experiment = hypothesis_framework.design_experiment(sample_hypothesis.hypothesis_id)
        hypothesis_framework.experiments[experiment.experiment_id] = experiment
        
        # Execute original experiment
        original_result = hypothesis_framework.execute_experiment(experiment.experiment_id)
        hypothesis_framework.results[experiment.experiment_id] = original_result
        
        # Validate reproducibility (with fewer replications for testing)
        reproducibility_analysis = hypothesis_framework.validate_reproducibility(
            experiment.experiment_id,
            num_replications=2
        )
        
        assert isinstance(reproducibility_analysis, dict)
        assert "reproducibility_score" in reproducibility_analysis
        assert "variance_analysis" in reproducibility_analysis
        assert "conclusions" in reproducibility_analysis
    
    def test_research_report_generation(self, hypothesis_framework, sample_hypothesis, temp_project_dir):
        """Test research report generation"""
        # Setup and execute experiment
        hypothesis_framework.hypotheses[sample_hypothesis.hypothesis_id] = sample_hypothesis
        experiment = hypothesis_framework.design_experiment(sample_hypothesis.hypothesis_id)
        hypothesis_framework.experiments[experiment.experiment_id] = experiment
        result = hypothesis_framework.execute_experiment(experiment.experiment_id)
        hypothesis_framework.results[experiment.experiment_id] = result
        
        # Generate report
        report_path = hypothesis_framework.generate_research_report(
            [experiment.experiment_id],
            output_format="markdown"
        )
        
        assert Path(report_path).exists()
        assert Path(report_path).suffix == ".md"
        
        # Check report content
        with open(report_path) as f:
            content = f.read()
            assert "Research Report" in content
            assert sample_hypothesis.title in content
    
    def test_hypothesis_generator(self, hypothesis_framework):
        """Test hypothesis generator component"""
        generator = hypothesis_framework.hypothesis_generator
        
        domain_knowledge = {"baselines": {"accuracy": 0.8}}
        research_gap = "Improve model accuracy"
        
        hypotheses = generator.generate_for_gap(research_gap, domain_knowledge)
        
        assert len(hypotheses) > 0
        assert all(isinstance(h, Hypothesis) for h in hypotheses)
        assert all(h.hypothesis_id is not None for h in hypotheses)
    
    def test_experiment_designer(self, hypothesis_framework, sample_hypothesis):
        """Test experiment designer component"""
        designer = hypothesis_framework.experiment_designer
        
        experiment = designer.design_experiment(sample_hypothesis)
        
        assert isinstance(experiment, ExperimentDesign)
        assert experiment.hypothesis == sample_hypothesis
        assert experiment.sample_size >= sample_hypothesis.minimum_sample_size
        assert len(experiment.dependent_variables) > 0
    
    def test_result_interpreter(self, hypothesis_framework, sample_hypothesis):
        """Test result interpreter component"""
        interpreter = hypothesis_framework.result_interpreter
        
        # Create mock experiment result
        result = ExperimentResult(
            experiment_id="test_exp",
            hypothesis_id=sample_hypothesis.hypothesis_id,
            status=ExperimentStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time() + 10,
            baseline_metrics={"accuracy": 0.80},
            proposed_metrics={"accuracy": 0.85},
            statistical_results={},
            effect_size=0.5,
            confidence_interval=(0.02, 0.08),
            p_value=0.01,
            conclusion=""
        )
        
        conclusion = interpreter.interpret_results(result, sample_hypothesis)
        
        assert isinstance(conclusion, str)
        assert len(conclusion) > 0
        assert any(keyword in conclusion.lower() for keyword in ["accept", "reject", "fail"])


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration tests for complete SDLC scenarios"""
    
    @pytest.fixture
    def temp_project_dir(self):
        """Create temporary project directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create realistic project structure
            (project_path / "spatial_omics_gfm").mkdir()
            (project_path / "tests").mkdir()
            (project_path / "docs").mkdir()
            
            # Create pyproject.toml
            pyproject_content = """
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "spatial-omics-gfm"
version = "1.0.0"
dependencies = ["torch", "numpy"]
"""
            (project_path / "pyproject.toml").write_text(pyproject_content)
            
            yield project_path
    
    @pytest.mark.asyncio
    async def test_complete_autonomous_sdlc(self, temp_project_dir):
        """Test complete autonomous SDLC execution"""
        # Initialize SDLC executor
        executor = AutonomousSDLCExecutor(temp_project_dir)
        
        # Mock quality gates to always pass
        with patch.object(executor.quality_gates, 'execute_all_gates', return_value=True):
            # Execute full SDLC (limited phases for testing)
            config = executor.config.copy()
            config["phases"] = {
                "analysis": {"enabled": True, "timeout": 1},
                "generation_1": {"enabled": True, "timeout": 1},
                "generation_2": {"enabled": False},  # Skip for faster testing
                "generation_3": {"enabled": False},
                "testing": {"enabled": True, "timeout": 1},
                "deployment": {"enabled": False},
                "monitoring": {"enabled": False}
            }
            
            executor.config = config
            
            success = await executor.execute_full_sdlc()
            
            assert isinstance(success, bool)
            assert len(executor.phase_results) > 0
            
            # Check that report was generated
            report_file = temp_project_dir / "autonomous_sdlc_report.json"
            assert report_file.exists()
    
    def test_quality_gates_with_adaptive_learning(self, temp_project_dir):
        """Test integration between quality gates and adaptive learning"""
        # Create quality gates
        gates = ProgressiveQualityGates(temp_project_dir)
        
        # Create simple model for adaptive learning
        model = nn.Linear(10, 2)
        adaptive_system = AdaptiveLearningSystem(model)
        
        # Mock successful quality gates
        with patch.object(gates, 'execute_all_gates', return_value=True):
            # Execute quality gates
            gates_result = gates.execute_all_gates()
            
            # If quality gates pass, trigger adaptive learning
            if gates_result:
                # Create dummy data
                from torch.utils.data import DataLoader, TensorDataset
                inputs = torch.randn(50, 10)
                targets = torch.randint(0, 2, (50,))
                dataset = TensorDataset(inputs, targets)
                dataloader = DataLoader(dataset, batch_size=10)
                
                # Adapt model
                adaptation_event = adaptive_system.adapt_to_new_data(
                    dataloader,
                    LearningStrategy.ONLINE_LEARNING
                )
                
                assert adaptation_event.successful
                assert len(adaptive_system.adaptation_history) == 1
    
    def test_hypothesis_testing_with_quality_gates(self, temp_project_dir):
        """Test integration between hypothesis testing and quality gates"""
        # Create hypothesis testing framework
        framework = HypothesisTestingFramework(temp_project_dir)
        
        # Generate hypothesis
        domain_knowledge = {"baselines": {"accuracy": 0.8}}
        research_gaps = ["Improve model performance"]
        
        hypotheses = framework.generate_research_hypotheses(domain_knowledge, research_gaps)
        assert len(hypotheses) > 0
        
        # Design and execute experiment
        hypothesis = hypotheses[0]
        experiment = framework.design_experiment(hypothesis.hypothesis_id)
        result = framework.execute_experiment(experiment.experiment_id)
        
        # If experiment successful, run quality gates
        if result.status == ExperimentStatus.COMPLETED:
            gates = ProgressiveQualityGates(temp_project_dir)
            
            with patch.object(gates, 'execute_all_gates', return_value=True):
                gates_result = gates.execute_all_gates()
                assert gates_result is True
    
    @pytest.mark.asyncio
    async def test_end_to_end_research_workflow(self, temp_project_dir):
        """Test complete end-to-end research workflow"""
        # 1. Initialize all components
        sdlc_executor = AutonomousSDLCExecutor(temp_project_dir)
        hypothesis_framework = HypothesisTestingFramework(temp_project_dir)
        model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
        adaptive_system = AdaptiveLearningSystem(model)
        
        # 2. Generate research hypotheses
        domain_knowledge = {
            "baselines": {"accuracy": 0.75, "speed": 100},
            "current_challenges": ["scalability", "accuracy"]
        }
        research_gaps = ["Improve spatial analysis accuracy", "Optimize processing speed"]
        
        hypotheses = hypothesis_framework.generate_research_hypotheses(
            domain_knowledge, research_gaps
        )
        
        assert len(hypotheses) > 0
        
        # 3. Execute autonomous SDLC for each hypothesis
        for hypothesis in hypotheses[:1]:  # Test with first hypothesis only
            # Design experiment
            experiment = hypothesis_framework.design_experiment(hypothesis.hypothesis_id)
            
            # Execute experiment with quality gates
            with patch.object(sdlc_executor.quality_gates, 'execute_all_gates', return_value=True):
                experiment_result = hypothesis_framework.execute_experiment(experiment.experiment_id)
                
                # If experiment successful, run SDLC phases
                if experiment_result.status == ExperimentStatus.COMPLETED:
                    # Execute analysis phase
                    analysis_result = await sdlc_executor._execute_analysis()
                    assert analysis_result.status == ExecutionStatus.COMPLETED
                    
                    # Execute generation 1 phase
                    gen1_result = await sdlc_executor._execute_generation_1()
                    assert gen1_result.status == ExecutionStatus.COMPLETED
                    
                    # Adapt model based on results
                    from torch.utils.data import DataLoader, TensorDataset
                    inputs = torch.randn(30, 10)
                    targets = torch.randint(0, 2, (30,))
                    dataset = TensorDataset(inputs, targets)
                    dataloader = DataLoader(dataset, batch_size=8)
                    
                    adaptation_event = adaptive_system.adapt_to_new_data(
                        dataloader,
                        LearningStrategy.ONLINE_LEARNING
                    )
                    
                    assert adaptation_event.successful
        
        # 4. Generate comprehensive report
        experiment_ids = list(hypothesis_framework.results.keys())
        if experiment_ids:
            report_path = hypothesis_framework.generate_research_report(experiment_ids)
            assert Path(report_path).exists()
        
        # 5. Validate final quality
        final_quality_result = sdlc_executor.quality_gates.execute_all_gates()
        assert isinstance(final_quality_result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])