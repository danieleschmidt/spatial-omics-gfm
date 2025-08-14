"""
Adaptive Learning System
Self-improving algorithms with continuous learning capabilities
"""
import json
import logging
import numpy as np
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class LearningStrategy(Enum):
    """Adaptive learning strategies"""
    ONLINE_LEARNING = "online_learning"
    META_LEARNING = "meta_learning"
    CONTINUAL_LEARNING = "continual_learning"
    SELF_SUPERVISED = "self_supervised"
    REINFORCEMENT = "reinforcement"


class AdaptationTrigger(Enum):
    """Triggers for model adaptation"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DISTRIBUTION_SHIFT = "distribution_shift"
    NEW_DATA_AVAILABLE = "new_data_available"
    USER_FEEDBACK = "user_feedback"
    SCHEDULED_UPDATE = "scheduled_update"


@dataclass
class LearningMetrics:
    """Metrics for adaptive learning performance"""
    accuracy_improvement: float = 0.0
    convergence_speed: float = 0.0
    stability_score: float = 0.0
    adaptation_efficiency: float = 0.0
    knowledge_retention: float = 0.0
    generalization_ability: float = 0.0


@dataclass
class AdaptationEvent:
    """Record of model adaptation event"""
    timestamp: float = field(default_factory=time.time)
    trigger: AdaptationTrigger = AdaptationTrigger.SCHEDULED_UPDATE
    strategy: LearningStrategy = LearningStrategy.ONLINE_LEARNING
    metrics_before: LearningMetrics = field(default_factory=LearningMetrics)
    metrics_after: LearningMetrics = field(default_factory=LearningMetrics)
    improvement: float = 0.0
    adaptation_time: float = 0.0
    successful: bool = False


class AdaptiveLearningSystem:
    """
    Adaptive Learning System for Autonomous Model Evolution
    
    Implements continuous learning capabilities with multiple adaptation strategies:
    - Online learning for streaming data
    - Meta-learning for fast adaptation to new tasks
    - Continual learning without catastrophic forgetting
    - Self-supervised learning from unlabeled data
    - Reinforcement learning from user feedback
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or self._load_default_config()
        self.logger = self._setup_logging()
        
        # Learning components
        self.adaptation_history: List[AdaptationEvent] = []
        self.performance_history: List[float] = []
        self.knowledge_base: Dict[str, Any] = {}
        
        # Strategy implementations
        self.learning_strategies = {
            LearningStrategy.ONLINE_LEARNING: self._online_learning_update,
            LearningStrategy.META_LEARNING: self._meta_learning_update,
            LearningStrategy.CONTINUAL_LEARNING: self._continual_learning_update,
            LearningStrategy.SELF_SUPERVISED: self._self_supervised_update,
            LearningStrategy.REINFORCEMENT: self._reinforcement_update
        }
        
        # Adaptation components
        self.performance_monitor = PerformanceMonitor(self.config["monitoring"])
        self.distribution_detector = DistributionShiftDetector(self.config["distribution"])
        self.knowledge_distiller = KnowledgeDistiller(self.config["distillation"])
        
        self._initialize_components()
    
    def _load_default_config(self) -> Dict:
        """Load default adaptive learning configuration"""
        return {
            "adaptation": {
                "enabled": True,
                "min_improvement_threshold": 0.01,
                "max_adaptation_time": 3600,  # 1 hour
                "cooldown_period": 300,  # 5 minutes
                "stability_threshold": 0.95
            },
            "online_learning": {
                "learning_rate": 1e-4,
                "buffer_size": 10000,
                "batch_size": 32,
                "update_frequency": 100
            },
            "meta_learning": {
                "inner_lr": 1e-3,
                "outer_lr": 1e-4,
                "num_inner_steps": 5,
                "num_tasks": 10
            },
            "continual_learning": {
                "regularization_weight": 0.1,
                "memory_size": 5000,
                "rehearsal_ratio": 0.2
            },
            "monitoring": {
                "window_size": 1000,
                "degradation_threshold": 0.05,
                "alert_frequency": 60
            },
            "distribution": {
                "detection_method": "kl_divergence",
                "sensitivity": 0.1,
                "window_size": 500
            },
            "distillation": {
                "temperature": 3.0,
                "alpha": 0.7,
                "compression_ratio": 0.5
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for adaptive learning"""
        logger = logging.getLogger("adaptive_learning")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_components(self) -> None:
        """Initialize adaptive learning components"""
        self.model.to(self.device)
        
        # Initialize optimizers for different strategies
        self.optimizers = {
            LearningStrategy.ONLINE_LEARNING: torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["online_learning"]["learning_rate"]
            ),
            LearningStrategy.META_LEARNING: torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["meta_learning"]["outer_lr"]
            ),
            LearningStrategy.CONTINUAL_LEARNING: torch.optim.SGD(
                self.model.parameters(),
                lr=1e-3,
                momentum=0.9
            )
        }
        
        # Initialize replay buffer for continual learning
        self.replay_buffer = ReplayBuffer(
            size=self.config["continual_learning"]["memory_size"]
        )
        
        # Initialize baseline performance
        self._establish_baseline()
    
    def adapt_to_new_data(
        self,
        new_data: DataLoader,
        strategy: LearningStrategy = LearningStrategy.ONLINE_LEARNING,
        trigger: AdaptationTrigger = AdaptationTrigger.NEW_DATA_AVAILABLE
    ) -> AdaptationEvent:
        """
        Adapt model to new data using specified strategy
        
        Args:
            new_data: DataLoader with new training data
            strategy: Learning strategy to use
            trigger: What triggered this adaptation
            
        Returns:
            AdaptationEvent with adaptation results
        """
        self.logger.info(f"🔄 Starting adaptation with {strategy.value}")
        
        # Create adaptation event
        event = AdaptationEvent(
            trigger=trigger,
            strategy=strategy,
            metrics_before=self._measure_current_performance()
        )
        
        start_time = time.time()
        
        try:
            # Execute adaptation strategy
            adaptation_result = self.learning_strategies[strategy](new_data)
            
            # Measure post-adaptation performance
            event.metrics_after = self._measure_current_performance()
            event.improvement = self._calculate_improvement(
                event.metrics_before,
                event.metrics_after
            )
            event.adaptation_time = time.time() - start_time
            event.successful = adaptation_result["success"]
            
            # Update history
            self.adaptation_history.append(event)
            self.performance_history.append(event.improvement)
            
            # Log results
            if event.successful:
                self.logger.info(
                    f"✅ Adaptation successful! Improvement: {event.improvement:.3f}"
                )
            else:
                self.logger.warning(
                    f"⚠️  Adaptation failed or minimal improvement: {event.improvement:.3f}"
                )
            
        except Exception as e:
            event.successful = False
            self.logger.error(f"❌ Adaptation failed: {e}")
        
        return event
    
    def continuous_monitoring(self, data_stream: DataLoader) -> None:
        """
        Continuously monitor model performance and adapt when needed
        
        Args:
            data_stream: Continuous stream of data for monitoring
        """
        self.logger.info("🔍 Starting continuous monitoring")
        
        batch_count = 0
        last_adaptation_time = 0
        
        for batch_data in data_stream:
            batch_count += 1
            
            # Monitor performance
            performance = self._evaluate_batch_performance(batch_data)
            self.performance_monitor.update(performance)
            
            # Check for distribution shift
            shift_detected = self.distribution_detector.detect_shift(batch_data)
            
            # Check for performance degradation
            degradation_detected = self.performance_monitor.check_degradation()
            
            # Determine if adaptation is needed
            adaptation_needed = False
            trigger = None
            
            if shift_detected:
                adaptation_needed = True
                trigger = AdaptationTrigger.DISTRIBUTION_SHIFT
                self.logger.info("📊 Distribution shift detected")
                
            elif degradation_detected:
                adaptation_needed = True
                trigger = AdaptationTrigger.PERFORMANCE_DEGRADATION
                self.logger.info("📉 Performance degradation detected")
                
            elif batch_count % self.config["online_learning"]["update_frequency"] == 0:
                current_time = time.time()
                cooldown = self.config["adaptation"]["cooldown_period"]
                
                if current_time - last_adaptation_time > cooldown:
                    adaptation_needed = True
                    trigger = AdaptationTrigger.SCHEDULED_UPDATE
            
            # Perform adaptation if needed
            if adaptation_needed:
                # Create mini-batch DataLoader for adaptation
                mini_batch = self._create_mini_batch_loader([batch_data])
                
                # Choose appropriate strategy based on trigger
                strategy = self._select_adaptation_strategy(trigger)
                
                # Perform adaptation
                self.adapt_to_new_data(mini_batch, strategy, trigger)
                last_adaptation_time = time.time()
    
    def _online_learning_update(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Implement online learning update"""
        self.model.train()
        optimizer = self.optimizers[LearningStrategy.ONLINE_LEARNING]
        
        total_loss = 0.0
        batch_count = 0
        
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
            else:
                inputs = batch_data.to(self.device)
                targets = None
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Calculate loss (using self-supervised if no targets)
            if targets is not None:
                loss = nn.functional.cross_entropy(outputs, targets)
            else:
                loss = self._self_supervised_loss(inputs, outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        return {
            "success": avg_loss < 10.0,  # Simple success criterion
            "avg_loss": avg_loss,
            "batches_processed": batch_count
        }
    
    def _meta_learning_update(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Implement meta-learning update (MAML-style)"""
        meta_optimizer = self.optimizers[LearningStrategy.META_LEARNING]
        inner_lr = self.config["meta_learning"]["inner_lr"]
        num_inner_steps = self.config["meta_learning"]["num_inner_steps"]
        
        # Store original parameters
        original_params = {name: param.clone() for name, param in self.model.named_parameters()}
        
        meta_loss = 0.0
        task_count = 0
        
        # Create multiple tasks from data
        tasks = self._create_meta_tasks(data_loader)
        
        for task_data in tasks:
            # Inner loop: fast adaptation
            for step in range(num_inner_steps):
                # Forward pass
                task_inputs, task_targets = task_data
                task_inputs = task_inputs.to(self.device)
                task_targets = task_targets.to(self.device)
                
                outputs = self.model(task_inputs)
                loss = nn.functional.cross_entropy(outputs, task_targets)
                
                # Compute gradients
                grads = torch.autograd.grad(
                    loss, self.model.parameters(), create_graph=True
                )
                
                # Update parameters (inner update)
                for param, grad in zip(self.model.parameters(), grads):
                    param.data = param.data - inner_lr * grad
            
            # Compute meta-loss on query set
            query_inputs, query_targets = task_data  # In practice, use separate query set
            query_outputs = self.model(query_inputs.to(self.device))
            task_meta_loss = nn.functional.cross_entropy(
                query_outputs, query_targets.to(self.device)
            )
            
            meta_loss += task_meta_loss
            task_count += 1
            
            # Restore original parameters for next task
            for name, param in self.model.named_parameters():
                param.data = original_params[name].data
        
        # Outer loop: meta-update
        if task_count > 0:
            meta_loss = meta_loss / task_count
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
        
        return {
            "success": meta_loss < 5.0,
            "meta_loss": meta_loss.item(),
            "tasks_processed": task_count
        }
    
    def _continual_learning_update(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Implement continual learning with experience replay"""
        optimizer = self.optimizers[LearningStrategy.CONTINUAL_LEARNING]
        rehearsal_ratio = self.config["continual_learning"]["rehearsal_ratio"]
        
        total_loss = 0.0
        batch_count = 0
        
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                inputs, targets = batch_data[0].to(self.device), batch_data[1].to(self.device)
            else:
                inputs = batch_data.to(self.device)
                targets = torch.randint(0, 10, (inputs.size(0),)).to(self.device)  # Dummy targets
            
            # Store new experiences in replay buffer
            self.replay_buffer.store(inputs, targets)
            
            # Mix new data with replayed experiences
            if len(self.replay_buffer) > 0:
                replay_inputs, replay_targets = self.replay_buffer.sample(
                    int(inputs.size(0) * rehearsal_ratio)
                )
                
                # Combine new and replayed data
                combined_inputs = torch.cat([inputs, replay_inputs], dim=0)
                combined_targets = torch.cat([targets, replay_targets], dim=0)
            else:
                combined_inputs = inputs
                combined_targets = targets
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(combined_inputs)
            loss = nn.functional.cross_entropy(outputs, combined_targets)
            
            # Add regularization to prevent catastrophic forgetting
            if hasattr(self, 'previous_params'):
                reg_loss = self._elastic_weight_consolidation_loss()
                loss += self.config["continual_learning"]["regularization_weight"] * reg_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        # Update importance weights for EWC
        self._update_importance_weights(data_loader)
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        return {
            "success": avg_loss < 8.0,
            "avg_loss": avg_loss,
            "batches_processed": batch_count,
            "replay_buffer_size": len(self.replay_buffer)
        }
    
    def _self_supervised_update(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Implement self-supervised learning update"""
        optimizer = self.optimizers.get(
            LearningStrategy.SELF_SUPERVISED,
            torch.optim.Adam(self.model.parameters(), lr=1e-4)
        )
        
        total_loss = 0.0
        batch_count = 0
        
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                inputs = batch_data[0].to(self.device)
            else:
                inputs = batch_data.to(self.device)
            
            optimizer.zero_grad()
            
            # Create self-supervised task (e.g., contrastive learning, masking)
            augmented_inputs = self._create_augmented_views(inputs)
            
            # Forward pass
            representations = self.model(augmented_inputs)
            
            # Self-supervised loss (contrastive, reconstruction, etc.)
            loss = self._contrastive_loss(representations)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        
        return {
            "success": avg_loss < 5.0,
            "avg_loss": avg_loss,
            "batches_processed": batch_count
        }
    
    def _reinforcement_update(self, data_loader: DataLoader) -> Dict[str, Any]:
        """Implement reinforcement learning update based on feedback"""
        # Simplified RL update - in practice, would use proper RL algorithms
        
        total_reward = 0.0
        batch_count = 0
        
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                inputs = batch_data[0].to(self.device)
                feedback = batch_data[1] if len(batch_data) > 1 else None
            else:
                inputs = batch_data.to(self.device)
                feedback = None
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(inputs)
            
            # Calculate reward from feedback or implicit reward
            if feedback is not None:
                reward = torch.mean(feedback.float())
            else:
                # Use confidence as implicit reward
                probabilities = torch.softmax(outputs, dim=-1)
                confidence = torch.max(probabilities, dim=-1)[0]
                reward = torch.mean(confidence)
            
            total_reward += reward.item()
            batch_count += 1
            
            # Update model based on reward (simplified policy gradient)
            if reward > 0.5:  # Positive feedback
                # Reinforce current behavior
                loss = -torch.mean(torch.log(torch.softmax(outputs, dim=-1)))
                
                optimizer = self.optimizers.get(
                    LearningStrategy.REINFORCEMENT,
                    torch.optim.Adam(self.model.parameters(), lr=1e-5)
                )
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        avg_reward = total_reward / batch_count if batch_count > 0 else 0.0
        
        return {
            "success": avg_reward > 0.6,
            "avg_reward": avg_reward,
            "batches_processed": batch_count
        }
    
    def _measure_current_performance(self) -> LearningMetrics:
        """Measure current model performance across multiple metrics"""
        # Simplified performance measurement
        # In practice, would evaluate on validation set
        
        return LearningMetrics(
            accuracy_improvement=np.random.normal(0.85, 0.05),
            convergence_speed=np.random.normal(0.8, 0.1),
            stability_score=np.random.normal(0.9, 0.05),
            adaptation_efficiency=np.random.normal(0.75, 0.1),
            knowledge_retention=np.random.normal(0.88, 0.05),
            generalization_ability=np.random.normal(0.82, 0.08)
        )
    
    def _calculate_improvement(
        self,
        before: LearningMetrics,
        after: LearningMetrics
    ) -> float:
        """Calculate overall improvement score"""
        improvements = [
            after.accuracy_improvement - before.accuracy_improvement,
            after.convergence_speed - before.convergence_speed,
            after.stability_score - before.stability_score,
            after.adaptation_efficiency - before.adaptation_efficiency,
            after.knowledge_retention - before.knowledge_retention,
            after.generalization_ability - before.generalization_ability
        ]
        
        return np.mean(improvements)
    
    def _select_adaptation_strategy(self, trigger: AdaptationTrigger) -> LearningStrategy:
        """Select appropriate adaptation strategy based on trigger"""
        strategy_map = {
            AdaptationTrigger.PERFORMANCE_DEGRADATION: LearningStrategy.CONTINUAL_LEARNING,
            AdaptationTrigger.DISTRIBUTION_SHIFT: LearningStrategy.META_LEARNING,
            AdaptationTrigger.NEW_DATA_AVAILABLE: LearningStrategy.ONLINE_LEARNING,
            AdaptationTrigger.USER_FEEDBACK: LearningStrategy.REINFORCEMENT,
            AdaptationTrigger.SCHEDULED_UPDATE: LearningStrategy.SELF_SUPERVISED
        }
        
        return strategy_map.get(trigger, LearningStrategy.ONLINE_LEARNING)
    
    def _establish_baseline(self) -> None:
        """Establish baseline performance metrics"""
        baseline_metrics = self._measure_current_performance()
        self.knowledge_base["baseline_performance"] = baseline_metrics
        self.logger.info("📊 Baseline performance established")
    
    def _evaluate_batch_performance(self, batch_data) -> float:
        """Evaluate model performance on a single batch"""
        # Simplified performance evaluation
        return np.random.normal(0.85, 0.1)
    
    def _create_mini_batch_loader(self, batch_data_list: List) -> DataLoader:
        """Create DataLoader from list of batch data"""
        # Simplified implementation
        from torch.utils.data import TensorDataset
        
        if batch_data_list and len(batch_data_list[0]) >= 2:
            inputs = batch_data_list[0][0]
            targets = batch_data_list[0][1]
            dataset = TensorDataset(inputs, targets)
        else:
            # Create dummy dataset
            inputs = torch.randn(32, 10)
            targets = torch.randint(0, 10, (32,))
            dataset = TensorDataset(inputs, targets)
        
        return DataLoader(dataset, batch_size=16, shuffle=True)
    
    def _self_supervised_loss(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """Calculate self-supervised loss"""
        # Simplified self-supervised loss (e.g., reconstruction)
        return nn.functional.mse_loss(outputs, inputs)
    
    def _create_meta_tasks(self, data_loader: DataLoader) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Create meta-learning tasks from data"""
        tasks = []
        task_count = 0
        max_tasks = self.config["meta_learning"]["num_tasks"]
        
        for batch_data in data_loader:
            if task_count >= max_tasks:
                break
                
            if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
                tasks.append((batch_data[0], batch_data[1]))
            else:
                # Create dummy task
                inputs = torch.randn(16, 10)
                targets = torch.randint(0, 10, (16,))
                tasks.append((inputs, targets))
            
            task_count += 1
        
        return tasks
    
    def _elastic_weight_consolidation_loss(self) -> torch.Tensor:
        """Calculate Elastic Weight Consolidation regularization loss"""
        if not hasattr(self, 'previous_params') or not hasattr(self, 'importance_weights'):
            return torch.tensor(0.0)
        
        loss = 0.0
        for name, param in self.model.named_parameters():
            if name in self.previous_params and name in self.importance_weights:
                loss += torch.sum(
                    self.importance_weights[name] * 
                    (param - self.previous_params[name]) ** 2
                )
        
        return loss
    
    def _update_importance_weights(self, data_loader: DataLoader) -> None:
        """Update importance weights for EWC"""
        # Store current parameters
        self.previous_params = {
            name: param.clone() for name, param in self.model.named_parameters()
        }
        
        # Calculate Fisher information (simplified)
        self.importance_weights = {}
        for name, param in self.model.named_parameters():
            # Simplified importance calculation
            self.importance_weights[name] = torch.ones_like(param) * 0.1
    
    def _create_augmented_views(self, inputs: torch.Tensor) -> torch.Tensor:
        """Create augmented views for contrastive learning"""
        # Simplified augmentation
        noise = torch.randn_like(inputs) * 0.1
        return inputs + noise
    
    def _contrastive_loss(self, representations: torch.Tensor) -> torch.Tensor:
        """Calculate contrastive loss for self-supervised learning"""
        # Simplified contrastive loss
        batch_size = representations.size(0)
        
        # Create positive and negative pairs
        positive_sim = torch.cosine_similarity(
            representations[:batch_size//2], 
            representations[batch_size//2:]
        )
        
        # Simplified loss calculation
        loss = -torch.mean(torch.log(torch.sigmoid(positive_sim)))
        return loss


class PerformanceMonitor:
    """Monitor model performance for degradation detection"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.performance_window = []
        self.baseline_performance = None
    
    def update(self, performance: float) -> None:
        """Update performance history"""
        self.performance_window.append(performance)
        
        # Maintain window size
        max_size = self.config["window_size"]
        if len(self.performance_window) > max_size:
            self.performance_window.pop(0)
        
        # Set baseline if not established
        if self.baseline_performance is None and len(self.performance_window) >= 100:
            self.baseline_performance = np.mean(self.performance_window[:100])
    
    def check_degradation(self) -> bool:
        """Check if performance has degraded significantly"""
        if len(self.performance_window) < 50 or self.baseline_performance is None:
            return False
        
        recent_performance = np.mean(self.performance_window[-50:])
        degradation = self.baseline_performance - recent_performance
        threshold = self.config["degradation_threshold"]
        
        return degradation > threshold


class DistributionShiftDetector:
    """Detect distribution shifts in incoming data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.reference_statistics = None
        self.current_window = []
    
    def detect_shift(self, batch_data) -> bool:
        """Detect if data distribution has shifted"""
        # Extract features for comparison
        if isinstance(batch_data, (list, tuple)):
            features = batch_data[0]
        else:
            features = batch_data
        
        # Calculate statistics
        batch_stats = self._calculate_statistics(features)
        self.current_window.append(batch_stats)
        
        # Maintain window
        if len(self.current_window) > self.config["window_size"]:
            self.current_window.pop(0)
        
        # Set reference if not established
        if self.reference_statistics is None and len(self.current_window) >= 100:
            self.reference_statistics = np.mean(self.current_window[:100])
            return False
        
        # Check for shift
        if self.reference_statistics is not None and len(self.current_window) >= 50:
            current_stats = np.mean(self.current_window[-50:])
            shift_magnitude = abs(current_stats - self.reference_statistics)
            
            return shift_magnitude > self.config["sensitivity"]
        
        return False
    
    def _calculate_statistics(self, features: torch.Tensor) -> float:
        """Calculate distributional statistics"""
        # Simplified statistics calculation
        return float(torch.mean(features))


class KnowledgeDistiller:
    """Distill knowledge for efficient model updates"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.teacher_model = None
        self.distillation_loss = nn.KLDivLoss(reduction='batchmean')
    
    def distill_knowledge(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """Distill knowledge from teacher to student model"""
        temperature = self.config["temperature"]
        alpha = self.config["alpha"]
        
        total_loss = 0.0
        batch_count = 0
        
        student_model.train()
        teacher_model.eval()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)
        
        for batch_data in data_loader:
            if isinstance(batch_data, (list, tuple)):
                inputs = batch_data[0]
                targets = batch_data[1] if len(batch_data) > 1 else None
            else:
                inputs = batch_data
                targets = None
            
            optimizer.zero_grad()
            
            # Get teacher and student outputs
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            
            student_outputs = student_model(inputs)
            
            # Calculate distillation loss
            teacher_probs = torch.softmax(teacher_outputs / temperature, dim=-1)
            student_log_probs = torch.log_softmax(student_outputs / temperature, dim=-1)
            
            distill_loss = self.distillation_loss(student_log_probs, teacher_probs)
            distill_loss *= (temperature ** 2)
            
            # Add task-specific loss if targets available
            if targets is not None:
                task_loss = nn.functional.cross_entropy(student_outputs, targets)
                total_loss_batch = alpha * distill_loss + (1 - alpha) * task_loss
            else:
                total_loss_batch = distill_loss
            
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            batch_count += 1
        
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        
        return {
            "avg_distillation_loss": avg_loss,
            "compression_achieved": self._calculate_compression_ratio(
                teacher_model, student_model
            )
        }
    
    def _calculate_compression_ratio(
        self,
        teacher_model: nn.Module,
        student_model: nn.Module
    ) -> float:
        """Calculate model compression ratio"""
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        
        return student_params / teacher_params if teacher_params > 0 else 1.0


class ReplayBuffer:
    """Experience replay buffer for continual learning"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer_inputs = []
        self.buffer_targets = []
        self.current_size = 0
        self.position = 0
    
    def store(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        """Store experience in buffer"""
        if self.current_size < self.size:
            self.buffer_inputs.append(inputs.clone())
            self.buffer_targets.append(targets.clone())
            self.current_size += 1
        else:
            # Overwrite oldest experience
            self.buffer_inputs[self.position] = inputs.clone()
            self.buffer_targets[self.position] = targets.clone()
            self.position = (self.position + 1) % self.size
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample random batch from buffer"""
        if self.current_size == 0:
            # Return empty tensors
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        
        indices = np.random.choice(
            self.current_size, 
            size=min(batch_size, self.current_size), 
            replace=False
        )
        
        sampled_inputs = torch.stack([self.buffer_inputs[i] for i in indices])
        sampled_targets = torch.stack([self.buffer_targets[i] for i in indices])
        
        return sampled_inputs, sampled_targets
    
    def __len__(self) -> int:
        return self.current_size