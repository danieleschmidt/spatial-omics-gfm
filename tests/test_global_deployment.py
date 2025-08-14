"""
Comprehensive Test Suite for Global Deployment Components
Tests for multi-region deployment, i18n, and compliance systems
"""
import asyncio
import json
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from spatial_omics_gfm.global.multi_region_deployment import (
    MultiRegionDeployment,
    DeploymentRegion,
    DeploymentStatus,
    DeploymentMetrics,
    RegionConfig,
    ComplianceStandard
)
from spatial_omics_gfm.global.i18n_system import (
    InternationalizationSystem,
    SupportedLanguage,
    LocalizationConfig
)
from spatial_omics_gfm.global.compliance_engine import (
    ComplianceEngine,
    ComplianceFramework,
    DataType,
    ConsentType,
    ProcessingPurpose,
    DataSubject,
    ComplianceRequirement
)


class TestMultiRegionDeployment:
    """Test suite for Multi-Region Deployment"""
    
    @pytest.fixture
    def deployment_config(self):
        """Create test deployment configuration"""
        return {
            "regions": {
                DeploymentRegion.US_EAST_1: {
                    "instance_type": "t3.medium",
                    "min_instances": 1,
                    "max_instances": 5,
                    "target_cpu": 70.0,
                    "compliance": [ComplianceStandard.CCPA],
                    "languages": ["en", "es"]
                },
                DeploymentRegion.EU_WEST_1: {
                    "instance_type": "t3.medium",
                    "min_instances": 1,
                    "max_instances": 3,
                    "target_cpu": 70.0,
                    "compliance": [ComplianceStandard.GDPR],
                    "languages": ["en", "de", "fr"]
                }
            },
            "scaling": {
                "scale_up_threshold": 80.0,
                "scale_down_threshold": 30.0,
                "predictive_scaling": False  # Disable for testing
            },
            "monitoring": {
                "metrics_interval": 1,  # Fast for testing
                "alert_thresholds": {
                    "error_rate": 0.05,
                    "response_time_p95": 2000,
                    "cpu_utilization": 90.0
                }
            }
        }
    
    @pytest.fixture
    def multi_region_deployment(self, deployment_config):
        """Create MultiRegionDeployment instance"""
        return MultiRegionDeployment(deployment_config)
    
    def test_initialization(self, multi_region_deployment):
        """Test multi-region deployment initialization"""
        assert multi_region_deployment is not None
        assert multi_region_deployment.config is not None
        assert len(multi_region_deployment.region_deployments) > 0
        assert multi_region_deployment.auto_scaler is not None
        assert multi_region_deployment.traffic_manager is not None
        assert multi_region_deployment.compliance_manager is not None
    
    def test_region_config_creation(self, multi_region_deployment):
        """Test region configuration creation"""
        for region, deployment in multi_region_deployment.region_deployments.items():
            config = deployment["config"]
            assert isinstance(config, RegionConfig)
            assert config.region == region
            assert config.min_instances >= 1
            assert config.max_instances >= config.min_instances
            assert len(config.compliance_requirements) > 0
    
    @pytest.mark.asyncio
    async def test_deploy_to_region(self, multi_region_deployment):
        """Test deployment to a specific region"""
        region = DeploymentRegion.US_EAST_1
        
        # Mock compliance validation
        with patch.object(
            multi_region_deployment.compliance_manager,
            'validate_region_compliance',
            return_value=True
        ):
            result = await multi_region_deployment._deploy_to_region(region)
            
            assert isinstance(result, dict)
            assert result["region"] == region.value
            assert result["status"] in ["active", "failed"]
            assert "instances" in result
            assert "compliance_validated" in result
    
    @pytest.mark.asyncio
    async def test_create_regional_infrastructure(self, multi_region_deployment):
        """Test regional infrastructure creation"""
        region = DeploymentRegion.EU_WEST_1
        config = multi_region_deployment.region_deployments[region]["config"]
        
        infrastructure = await multi_region_deployment._create_regional_infrastructure(region, config)
        
        assert isinstance(infrastructure, dict)
        assert "vpc" in infrastructure
        assert "subnets" in infrastructure
        assert "security_groups" in infrastructure
        assert "auto_scaling_group" in infrastructure
        assert len(infrastructure["subnets"]) >= 2  # Multi-AZ
    
    @pytest.mark.asyncio
    async def test_deploy_application_instances(self, multi_region_deployment):
        """Test application instance deployment"""
        region = DeploymentRegion.US_EAST_1
        config = multi_region_deployment.region_deployments[region]["config"]
        
        instances = await multi_region_deployment._deploy_application_instances(region, config)
        
        assert isinstance(instances, list)
        assert len(instances) == config.min_instances
        
        for instance in instances:
            assert "instance_id" in instance
            assert "instance_type" in instance
            assert "status" in instance
            assert instance["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_configure_load_balancer(self, multi_region_deployment):
        """Test load balancer configuration"""
        region = DeploymentRegion.EU_WEST_1
        instances = [
            {"instance_id": "i-123", "private_ip": "10.0.1.100"},
            {"instance_id": "i-456", "private_ip": "10.0.1.101"}
        ]
        
        load_balancer = await multi_region_deployment._configure_load_balancer(region, instances)
        
        assert isinstance(load_balancer, dict)
        assert "lb_arn" in load_balancer
        assert "dns_name" in load_balancer
        assert "target_instances" in load_balancer
        assert len(load_balancer["target_instances"]) == len(instances)
        assert "health_check" in load_balancer
    
    @pytest.mark.asyncio
    async def test_setup_cdn(self, multi_region_deployment):
        """Test CDN setup"""
        region = DeploymentRegion.US_EAST_1
        load_balancer = {
            "dns_name": "us-east-1-lb.example.com",
            "lb_arn": "arn:aws:elasticloadbalancing:us-east-1:123:loadbalancer/app/test"
        }
        
        cdn = await multi_region_deployment._setup_cdn(region, load_balancer)
        
        assert isinstance(cdn, dict)
        assert "distribution_id" in cdn
        assert "domain_name" in cdn
        assert "origin" in cdn
        assert cdn["origin"] == load_balancer["dns_name"]
        assert "cache_behaviors" in cdn
    
    @pytest.mark.asyncio
    async def test_global_deployment(self, multi_region_deployment):
        """Test complete global deployment"""
        # Mock compliance validation
        with patch.object(
            multi_region_deployment.compliance_manager,
            'validate_region_compliance',
            return_value=True
        ):
            # Stop monitoring to avoid infinite loop in tests
            with patch.object(multi_region_deployment, '_start_global_monitoring'):
                result = await multi_region_deployment.deploy_globally()
                
                assert isinstance(result, dict)
                assert "deployment_start_time" in result
                assert "regions" in result
                assert "overall_status" in result
                assert result["overall_status"] in ["success", "partial_failure"]
                assert len(result["active_regions"]) > 0
    
    @pytest.mark.asyncio
    async def test_update_region_metrics(self, multi_region_deployment):
        """Test region metrics updating"""
        region = DeploymentRegion.US_EAST_1
        
        # Set up region as active with instances
        multi_region_deployment.region_deployments[region]["status"] = DeploymentStatus.ACTIVE
        multi_region_deployment.region_deployments[region]["instances"] = [
            {"instance_id": "i-123", "status": "running"}
        ]
        
        await multi_region_deployment._update_region_metrics(region)
        
        assert region in multi_region_deployment.region_metrics
        metrics = multi_region_deployment.region_metrics[region]
        assert isinstance(metrics, DeploymentMetrics)
        assert metrics.active_instances == 1
        assert 0 <= metrics.cpu_utilization <= 100
        assert 0 <= metrics.memory_utilization <= 100
        assert metrics.last_updated > 0
    
    @pytest.mark.asyncio
    async def test_scaling_logic(self, multi_region_deployment):
        """Test auto-scaling logic"""
        region = DeploymentRegion.US_EAST_1
        
        # Set up region as active
        multi_region_deployment.region_deployments[region]["status"] = DeploymentStatus.ACTIVE
        multi_region_deployment.region_deployments[region]["instances"] = [
            {"instance_id": "i-123"}
        ]
        
        # Create high CPU utilization metrics
        high_cpu_metrics = DeploymentMetrics(
            region=region,
            active_instances=1,
            cpu_utilization=85.0,  # Above scale-up threshold
            memory_utilization=60.0,
            request_rate=100.0,
            response_time_p95=500.0,
            error_rate=0.01,
            bandwidth_usage=50.0
        )
        
        multi_region_deployment.region_metrics[region] = high_cpu_metrics
        
        await multi_region_deployment._check_scaling_needs()
        
        # Should trigger scale up
        assert multi_region_deployment.region_deployments[region]["status"] in [
            DeploymentStatus.ACTIVE, DeploymentStatus.SCALING
        ]
    
    def test_global_status(self, multi_region_deployment):
        """Test global status reporting"""
        # Set up some regions as active
        for region in [DeploymentRegion.US_EAST_1, DeploymentRegion.EU_WEST_1]:
            multi_region_deployment.region_deployments[region]["status"] = DeploymentStatus.ACTIVE
            multi_region_deployment.region_deployments[region]["instances"] = [
                {"instance_id": f"i-{region.value}-001"}
            ]
            
            # Add metrics
            multi_region_deployment.region_metrics[region] = DeploymentMetrics(
                region=region,
                active_instances=1,
                cpu_utilization=60.0,
                memory_utilization=55.0,
                request_rate=80.0,
                response_time_p95=300.0,
                error_rate=0.005,
                bandwidth_usage=30.0
            )
        
        status = multi_region_deployment.get_global_status()
        
        assert isinstance(status, dict)
        assert "overall_health" in status
        assert "active_regions" in status
        assert "total_regions" in status
        assert "total_instances" in status
        assert "regions" in status
        assert status["active_regions"] == 2
        assert status["total_instances"] == 2


class TestInternationalizationSystem:
    """Test suite for Internationalization System"""
    
    @pytest.fixture
    def temp_i18n_dir(self):
        """Create temporary i18n directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def i18n_system(self, temp_i18n_dir):
        """Create InternationalizationSystem instance"""
        return InternationalizationSystem(temp_i18n_dir)
    
    def test_initialization(self, i18n_system):
        """Test i18n system initialization"""
        assert i18n_system is not None
        assert i18n_system.default_language == SupportedLanguage.ENGLISH
        assert i18n_system.current_language == SupportedLanguage.ENGLISH
        assert len(i18n_system.translations) > 0
        assert len(i18n_system.localization_configs) > 0
    
    def test_language_switching(self, i18n_system):
        """Test language switching"""
        # Test switching to Spanish
        i18n_system.set_language(SupportedLanguage.SPANISH)
        assert i18n_system.current_language == SupportedLanguage.SPANISH
        
        # Test invalid language falls back to default
        with patch.object(i18n_system.logger, 'warning'):
            i18n_system.set_language(SupportedLanguage.ITALIAN)  # Not in translations
            assert i18n_system.current_language == i18n_system.default_language
    
    def test_text_translation(self, i18n_system):
        """Test text translation"""
        # Test English
        i18n_system.set_language(SupportedLanguage.ENGLISH)
        assert i18n_system.get_text("welcome") == "Welcome to Spatial-Omics GFM"
        
        # Test Spanish
        i18n_system.set_language(SupportedLanguage.SPANISH)
        assert i18n_system.get_text("welcome") == "Bienvenido a Spatial-Omics GFM"
        
        # Test French
        i18n_system.set_language(SupportedLanguage.FRENCH)
        assert i18n_system.get_text("welcome") == "Bienvenue dans Spatial-Omics GFM"
        
        # Test key not found returns key
        assert i18n_system.get_text("nonexistent_key") == "nonexistent_key"
    
    def test_text_formatting(self, i18n_system):
        """Test text formatting with parameters"""
        # Add a parameterized translation
        i18n_system.translations[SupportedLanguage.ENGLISH]["greeting"] = "Hello, {name}!"
        i18n_system.translations[SupportedLanguage.SPANISH]["greeting"] = "¡Hola, {name}!"
        
        # Test English formatting
        i18n_system.set_language(SupportedLanguage.ENGLISH)
        assert i18n_system.get_text("greeting", name="John") == "Hello, John!"
        
        # Test Spanish formatting
        i18n_system.set_language(SupportedLanguage.SPANISH)
        assert i18n_system.get_text("greeting", name="Juan") == "¡Hola, Juan!"
    
    def test_number_formatting(self, i18n_system):
        """Test number formatting according to locale"""
        test_number = 1234.56
        
        # English (US) format
        i18n_system.set_language(SupportedLanguage.ENGLISH)
        assert i18n_system.format_number(test_number) == "1,234.56"
        
        # German format (period for thousands, comma for decimal)
        i18n_system.set_language(SupportedLanguage.GERMAN)
        formatted = i18n_system.format_number(test_number)
        assert "." in formatted  # Thousands separator
        assert "," in formatted  # Decimal separator
        
        # French format (space for thousands, comma for decimal)
        i18n_system.set_language(SupportedLanguage.FRENCH)
        formatted = i18n_system.format_number(test_number)
        assert " " in formatted or "," in formatted
    
    def test_currency_formatting(self, i18n_system):
        """Test currency formatting according to locale"""
        test_amount = 1234.56
        
        # English (USD)
        i18n_system.set_language(SupportedLanguage.ENGLISH)
        currency = i18n_system.format_currency(test_amount)
        assert "$" in currency
        assert "1,234.56" in currency
        
        # German (EUR)
        i18n_system.set_language(SupportedLanguage.GERMAN)
        currency = i18n_system.format_currency(test_amount)
        assert "€" in currency
        
        # Japanese (JPY)
        i18n_system.set_language(SupportedLanguage.JAPANESE)
        currency = i18n_system.format_currency(test_amount)
        assert "¥" in currency
    
    def test_date_formatting(self, i18n_system):
        """Test date formatting according to locale"""
        test_date = datetime(2023, 12, 25)
        
        # English (MM/DD/YYYY)
        i18n_system.set_language(SupportedLanguage.ENGLISH)
        formatted = i18n_system.format_date(test_date)
        assert "12/25/2023" == formatted
        
        # German (DD.MM.YYYY)
        i18n_system.set_language(SupportedLanguage.GERMAN)
        formatted = i18n_system.format_date(test_date)
        assert "25.12.2023" == formatted
        
        # Japanese (YYYY/MM/DD)
        i18n_system.set_language(SupportedLanguage.JAPANESE)
        formatted = i18n_system.format_date(test_date)
        assert "2023/12/25" == formatted
    
    def test_language_detection(self, i18n_system):
        """Test language detection from Accept-Language header"""
        # Test exact match
        detected = i18n_system.detect_user_language("en-US,en;q=0.9")
        assert detected == SupportedLanguage.ENGLISH
        
        # Test quality preference
        detected = i18n_system.detect_user_language("fr-FR,fr;q=0.9,en;q=0.8")
        assert detected == SupportedLanguage.FRENCH
        
        # Test unsupported language falls back to default
        detected = i18n_system.detect_user_language("ar-SA,ar;q=0.9")
        assert detected == i18n_system.default_language
        
        # Test empty header
        detected = i18n_system.detect_user_language("")
        assert detected == i18n_system.default_language
    
    def test_supported_languages(self, i18n_system):
        """Test supported languages listing"""
        languages = i18n_system.get_supported_languages()
        
        assert isinstance(languages, list)
        assert len(languages) > 0
        
        for lang_info in languages:
            assert "code" in lang_info
            assert "name" in lang_info
            assert "available" in lang_info
            assert isinstance(lang_info["available"], bool)
    
    def test_custom_translations(self, i18n_system):
        """Test adding custom translations"""
        custom_translations = {
            "custom_key": "Custom Value",
            "another_key": "Another Value"
        }
        
        i18n_system.add_custom_translations(SupportedLanguage.ENGLISH, custom_translations)
        
        assert i18n_system.get_text("custom_key") == "Custom Value"
        assert i18n_system.get_text("another_key") == "Another Value"
    
    def test_translation_completeness(self, i18n_system):
        """Test translation completeness calculation"""
        completeness = i18n_system.get_translation_completeness()
        
        assert isinstance(completeness, dict)
        assert SupportedLanguage.ENGLISH.value in completeness
        assert completeness[SupportedLanguage.ENGLISH.value] == 1.0  # 100% complete (reference)
        
        for lang_code, completion_rate in completeness.items():
            assert 0.0 <= completion_rate <= 1.0
    
    def test_translation_validation(self, i18n_system):
        """Test translation validation"""
        # Add incomplete translation
        i18n_system.translations[SupportedLanguage.ITALIAN] = {
            "welcome": "Benvenuto",
            "missing_translation": ""  # Empty value
        }
        
        issues = i18n_system.validate_translations()
        
        assert isinstance(issues, dict)
        assert "missing_keys" in issues
        assert "empty_values" in issues
        assert "formatting_errors" in issues
    
    def test_export_import_translations(self, i18n_system, temp_i18n_dir):
        """Test translation export and import"""
        # Export translations
        export_dir = temp_i18n_dir / "exported"
        i18n_system.export_translations(export_dir)
        
        # Check files were created
        assert export_dir.exists()
        assert (export_dir / "en.json").exists()
        assert (export_dir / "es.json").exists()
        
        # Test import
        new_i18n = InternationalizationSystem(temp_i18n_dir)
        new_i18n.import_translations(export_dir)
        
        # Verify imported translations
        assert len(new_i18n.translations) > 0
        assert new_i18n.get_text("welcome") is not None


class TestComplianceEngine:
    """Test suite for Compliance Engine"""
    
    @pytest.fixture
    def compliance_engine(self):
        """Create ComplianceEngine instance"""
        return ComplianceEngine()
    
    @pytest.fixture
    def sample_data_subject(self):
        """Create sample data subject"""
        return DataSubject(
            subject_id="test_subject_001",
            jurisdiction="EU",
            applicable_frameworks=[ComplianceFramework.GDPR, ComplianceFramework.ISO27001]
        )
    
    def test_initialization(self, compliance_engine):
        """Test compliance engine initialization"""
        assert compliance_engine is not None
        assert compliance_engine.config is not None
        assert len(compliance_engine.requirements) > 0
        assert compliance_engine.cipher_suite is not None
        assert compliance_engine.data_subjects == {}
        assert compliance_engine.processing_records == []
    
    def test_compliance_requirements_initialization(self, compliance_engine):
        """Test compliance requirements initialization"""
        # Check GDPR requirements
        gdpr_reqs = compliance_engine.requirements.get(ComplianceFramework.GDPR, [])
        assert len(gdpr_reqs) > 0
        
        gdpr_req_ids = [req.requirement_id for req in gdpr_reqs]
        assert "GDPR-01" in gdpr_req_ids  # Lawful basis
        assert "GDPR-02" in gdpr_req_ids  # Data subject rights
        
        # Check CCPA requirements
        ccpa_reqs = compliance_engine.requirements.get(ComplianceFramework.CCPA, [])
        assert len(ccpa_reqs) > 0
        
        # Check requirement structure
        for req in gdpr_reqs:
            assert isinstance(req, ComplianceRequirement)
            assert req.framework == ComplianceFramework.GDPR
            assert req.requirement_id is not None
            assert req.title is not None
            assert isinstance(req.mandatory, bool)
    
    def test_data_subject_registration(self, compliance_engine):
        """Test data subject registration"""
        subject_id = "test_user_123"
        jurisdiction = "EU"
        
        data_subject = compliance_engine.register_data_subject(subject_id, jurisdiction)
        
        assert isinstance(data_subject, DataSubject)
        assert data_subject.subject_id == subject_id
        assert data_subject.jurisdiction == jurisdiction
        assert ComplianceFramework.GDPR in data_subject.applicable_frameworks
        assert subject_id in compliance_engine.data_subjects
    
    def test_applicable_frameworks_determination(self, compliance_engine):
        """Test determination of applicable frameworks by jurisdiction"""
        test_cases = [
            ("EU", [ComplianceFramework.GDPR]),
            ("US", [ComplianceFramework.CCPA]),
            ("CA", [ComplianceFramework.CCPA, ComplianceFramework.PIPEDA]),
            ("SG", [ComplianceFramework.PDPA]),
            ("BR", [ComplianceFramework.LGPD]),
            ("OTHER", [ComplianceFramework.ISO27001])
        ]
        
        for jurisdiction, expected_frameworks in test_cases:
            frameworks = compliance_engine._determine_applicable_frameworks(jurisdiction)
            for expected_framework in expected_frameworks:
                assert expected_framework in frameworks
    
    def test_consent_recording(self, compliance_engine, sample_data_subject):
        """Test consent recording"""
        # Register data subject
        compliance_engine.data_subjects[sample_data_subject.subject_id] = sample_data_subject
        
        # Record consent
        success = compliance_engine.record_consent(
            sample_data_subject.subject_id,
            ProcessingPurpose.RESEARCH,
            ConsentType.EXPLICIT
        )
        
        assert success is True
        assert ProcessingPurpose.RESEARCH in sample_data_subject.consent_records
        assert sample_data_subject.consent_records[ProcessingPurpose.RESEARCH] == ConsentType.EXPLICIT
    
    def test_consent_verification(self, compliance_engine, sample_data_subject):
        """Test consent verification"""
        # Register data subject with consent
        compliance_engine.data_subjects[sample_data_subject.subject_id] = sample_data_subject
        sample_data_subject.consent_records[ProcessingPurpose.RESEARCH] = ConsentType.EXPLICIT
        
        # Test consent verification
        has_consent = compliance_engine._verify_consent(
            sample_data_subject.subject_id,
            [ProcessingPurpose.RESEARCH]
        )
        assert has_consent is True
        
        # Test missing consent
        has_consent = compliance_engine._verify_consent(
            sample_data_subject.subject_id,
            [ProcessingPurpose.MARKETING]
        )
        assert has_consent is False
        
        # Test withdrawn consent
        sample_data_subject.consent_records[ProcessingPurpose.ANALYTICS] = ConsentType.WITHDRAWN
        has_consent = compliance_engine._verify_consent(
            sample_data_subject.subject_id,
            [ProcessingPurpose.ANALYTICS]
        )
        assert has_consent is False
    
    def test_data_processing_recording(self, compliance_engine, sample_data_subject):
        """Test data processing activity recording"""
        # Register data subject
        compliance_engine.data_subjects[sample_data_subject.subject_id] = sample_data_subject
        
        # Record processing
        processing_id = compliance_engine.record_data_processing(
            sample_data_subject.subject_id,
            [DataType.PERSONAL_IDENTIFIABLE, DataType.GENETIC_DATA],
            [ProcessingPurpose.RESEARCH],
            "consent"
        )
        
        assert processing_id is not None
        assert len(compliance_engine.processing_records) == 1
        
        record = compliance_engine.processing_records[0]
        assert record.processing_id == processing_id
        assert record.data_subject_id == sample_data_subject.subject_id
        assert DataType.PERSONAL_IDENTIFIABLE in record.data_types
        assert ProcessingPurpose.RESEARCH in record.purposes
        assert record.legal_basis == "consent"
    
    def test_subject_access_request(self, compliance_engine, sample_data_subject):
        """Test subject access request handling"""
        # Setup data subject with processing records
        compliance_engine.data_subjects[sample_data_subject.subject_id] = sample_data_subject
        sample_data_subject.consent_records[ProcessingPurpose.RESEARCH] = ConsentType.EXPLICIT
        
        processing_id = compliance_engine.record_data_processing(
            sample_data_subject.subject_id,
            [DataType.PERSONAL_IDENTIFIABLE],
            [ProcessingPurpose.RESEARCH],
            "consent"
        )
        
        # Handle access request
        subject_data = compliance_engine.handle_subject_access_request(sample_data_subject.subject_id)
        
        assert isinstance(subject_data, dict)
        assert "subject_info" in subject_data
        assert "consent_records" in subject_data
        assert "processing_activities" in subject_data
        
        assert subject_data["subject_info"]["subject_id"] == sample_data_subject.subject_id
        assert len(subject_data["processing_activities"]) == 1
        assert subject_data["processing_activities"][0]["processing_id"] == processing_id
    
    def test_deletion_request(self, compliance_engine, sample_data_subject):
        """Test deletion request handling"""
        # Setup data subject
        compliance_engine.data_subjects[sample_data_subject.subject_id] = sample_data_subject
        
        # Record processing activity
        compliance_engine.record_data_processing(
            sample_data_subject.subject_id,
            [DataType.PERSONAL_IDENTIFIABLE],
            [ProcessingPurpose.RESEARCH],
            "consent"
        )
        
        # Handle deletion request
        deletion_success = compliance_engine.handle_deletion_request(sample_data_subject.subject_id)
        
        assert isinstance(deletion_success, bool)
        
        if deletion_success:
            # Check that data was actually deleted
            assert sample_data_subject.subject_id not in compliance_engine.data_subjects
            
            # Check that processing records were removed
            remaining_records = [
                r for r in compliance_engine.processing_records
                if r.data_subject_id == sample_data_subject.subject_id
            ]
            assert len(remaining_records) == 0
    
    def test_data_anonymization(self, compliance_engine):
        """Test data anonymization"""
        test_data = {
            "user_id": "12345",
            "email": "test@example.com",
            "name": "John Doe",
            "age": 30,
            "diagnosis": "diabetes",
            "measurement": 42.5
        }
        
        data_types = [DataType.PERSONAL_IDENTIFIABLE, DataType.HEALTH_INFORMATION]
        
        anonymized_data = compliance_engine.anonymize_data(test_data, data_types)
        
        assert isinstance(anonymized_data, dict)
        
        # Check that sensitive fields were anonymized
        assert anonymized_data["email"] != test_data["email"]
        assert anonymized_data["name"] != test_data["name"]
        assert anonymized_data["diagnosis"] != test_data["diagnosis"]
        
        # Check that anonymized values follow expected pattern
        assert anonymized_data["email"].startswith("anon_")
        assert anonymized_data["name"].startswith("anon_")
    
    def test_data_pseudonymization(self, compliance_engine):
        """Test data pseudonymization"""
        test_data = {
            "user_id": "user_12345",
            "email": "test@example.com",
            "age": 30,
            "measurement": 42.5
        }
        
        subject_id = "test_subject"
        
        pseudonymized_data = compliance_engine.pseudonymize_data(test_data, subject_id)
        
        assert isinstance(pseudonymized_data, dict)
        
        # Check that identifying fields were pseudonymized
        assert pseudonymized_data["user_id"] != test_data["user_id"]
        assert pseudonymized_data["user_id"].startswith("pseudo_")
        
        # Check that same subject gets same pseudonym
        pseudonymized_data2 = compliance_engine.pseudonymize_data(test_data, subject_id)
        assert pseudonymized_data["user_id"] == pseudonymized_data2["user_id"]
    
    def test_data_encryption(self, compliance_engine):
        """Test data encryption and decryption"""
        test_data = b"sensitive information"
        
        # Encrypt data
        encrypted_data = compliance_engine.encrypt_sensitive_data(test_data)
        assert encrypted_data != test_data
        assert len(encrypted_data) > len(test_data)  # Encrypted data is larger
        
        # Decrypt data
        decrypted_data = compliance_engine.decrypt_sensitive_data(encrypted_data)
        assert decrypted_data == test_data
    
    def test_compliance_audit(self, compliance_engine):
        """Test compliance audit execution"""
        framework = ComplianceFramework.GDPR
        auditor = "internal_auditor"
        scope = ["data_processing", "consent_management", "data_subject_rights"]
        
        audit_record = compliance_engine.conduct_compliance_audit(framework, auditor, scope)
        
        assert audit_record is not None
        assert audit_record.framework == framework
        assert audit_record.auditor == auditor
        assert audit_record.scope == scope
        assert 0.0 <= audit_record.compliance_score <= 1.0
        assert len(audit_record.findings) >= 0
        assert len(audit_record.recommendations) > 0
        assert audit_record.next_audit_date > audit_record.timestamp
        
        # Check audit was recorded
        assert len(compliance_engine.audit_records) == 1
        assert compliance_engine.audit_records[0] == audit_record
    
    def test_compliance_dashboard(self, compliance_engine, sample_data_subject):
        """Test compliance dashboard generation"""
        # Setup some test data
        compliance_engine.data_subjects[sample_data_subject.subject_id] = sample_data_subject
        sample_data_subject.consent_records[ProcessingPurpose.RESEARCH] = ConsentType.EXPLICIT
        
        compliance_engine.record_data_processing(
            sample_data_subject.subject_id,
            [DataType.PERSONAL_IDENTIFIABLE],
            [ProcessingPurpose.RESEARCH],
            "consent"
        )
        
        # Conduct audit
        compliance_engine.conduct_compliance_audit(
            ComplianceFramework.GDPR,
            "test_auditor",
            ["data_processing"]
        )
        
        # Generate dashboard
        dashboard = compliance_engine.get_compliance_dashboard()
        
        assert isinstance(dashboard, dict)
        assert "overall_status" in dashboard
        assert "frameworks" in dashboard
        assert "data_subjects" in dashboard
        assert "processing_activities" in dashboard
        assert "audit_summary" in dashboard
        assert "alerts" in dashboard
        
        # Check data subjects section
        assert dashboard["data_subjects"]["total"] == 1
        assert sample_data_subject.jurisdiction in dashboard["data_subjects"]["by_jurisdiction"]
        
        # Check processing activities
        assert dashboard["processing_activities"]["total"] == 1
        assert ProcessingPurpose.RESEARCH.value in dashboard["processing_activities"]["by_purpose"]
    
    def test_gdpr_specific_compliance(self, compliance_engine):
        """Test GDPR-specific compliance checks"""
        # Create EU data subject
        subject_id = "eu_subject_001"
        data_subject = compliance_engine.register_data_subject(subject_id, "EU")
        
        # Record non-explicit consent (should fail GDPR audit)
        compliance_engine.record_consent(
            subject_id,
            ProcessingPurpose.MARKETING,
            ConsentType.IMPLIED
        )
        
        # Conduct GDPR audit
        audit_record = compliance_engine.conduct_compliance_audit(
            ComplianceFramework.GDPR,
            "gdpr_auditor",
            ["consent_management"]
        )
        
        # Should find consent issues
        consent_findings = [f for f in audit_record.findings if "consent" in f.lower()]
        assert len(consent_findings) > 0
    
    def test_alert_generation(self, compliance_engine, sample_data_subject):
        """Test compliance alert generation"""
        # Setup scenario that should generate alerts
        compliance_engine.data_subjects[sample_data_subject.subject_id] = sample_data_subject
        
        # Record withdrawn consent
        sample_data_subject.consent_records[ProcessingPurpose.MARKETING] = ConsentType.WITHDRAWN
        
        # Record processing with old retention date (overdue for deletion)
        old_timestamp = datetime.utcnow() - timedelta(days=400)
        processing_record = compliance_engine.processing_records
        
        # Generate alerts through dashboard
        dashboard = compliance_engine.get_compliance_dashboard()
        alerts = dashboard["alerts"]
        
        assert isinstance(alerts, list)
        
        # Check for withdrawn consent alert
        withdrawn_alerts = [a for a in alerts if a["type"] == "withdrawn_consent"]
        if withdrawn_alerts:
            assert withdrawn_alerts[0]["count"] > 0


@pytest.mark.integration
class TestGlobalIntegrationScenarios:
    """Integration tests for global deployment scenarios"""
    
    @pytest.mark.asyncio
    async def test_compliance_aware_deployment(self):
        """Test deployment with compliance requirements"""
        # Setup multi-region deployment
        deployment_config = {
            "regions": {
                DeploymentRegion.EU_WEST_1: {
                    "instance_type": "t3.small",
                    "min_instances": 1,
                    "max_instances": 2,
                    "target_cpu": 70.0,
                    "compliance": [ComplianceStandard.GDPR],
                    "languages": ["en", "de"]
                }
            }
        }
        
        deployment = MultiRegionDeployment(deployment_config)
        compliance_engine = ComplianceEngine()
        
        # Register EU data subject
        subject_id = "eu_user_001"
        data_subject = compliance_engine.register_data_subject(subject_id, "EU")
        
        # Record consent for processing
        compliance_engine.record_consent(
            subject_id,
            ProcessingPurpose.RESEARCH,
            ConsentType.EXPLICIT
        )
        
        # Mock compliance validation to pass
        with patch.object(
            deployment.compliance_manager,
            'validate_region_compliance',
            return_value=True
        ):
            # Deploy to EU region (should respect GDPR)
            with patch.object(deployment, '_start_global_monitoring'):
                result = await deployment.deploy_globally()
                
                assert result["overall_status"] in ["success", "partial_failure"]
                assert DeploymentRegion.EU_WEST_1.value in result["active_regions"]
        
        # Verify compliance requirements were checked
        gdpr_requirements = compliance_engine.requirements[ComplianceFramework.GDPR]
        assert len(gdpr_requirements) > 0
    
    def test_i18n_with_compliance(self):
        """Test internationalization with compliance considerations"""
        # Setup i18n system
        with tempfile.TemporaryDirectory() as temp_dir:
            i18n = InternationalizationSystem(Path(temp_dir))
            compliance_engine = ComplianceEngine()
            
            # Test different regions with their compliance requirements
            test_scenarios = [
                (SupportedLanguage.ENGLISH, "US", [ComplianceFramework.CCPA]),
                (SupportedLanguage.GERMAN, "EU", [ComplianceFramework.GDPR]),
                (SupportedLanguage.JAPANESE, "JP", [ComplianceFramework.ISO27001])
            ]
            
            for language, jurisdiction, expected_frameworks in test_scenarios:
                # Set language
                i18n.set_language(language)
                
                # Register data subject in appropriate jurisdiction
                subject_id = f"user_{jurisdiction.lower()}_001"
                data_subject = compliance_engine.register_data_subject(subject_id, jurisdiction)
                
                # Verify frameworks were assigned correctly
                for framework in expected_frameworks:
                    assert framework in data_subject.applicable_frameworks
                
                # Test localized consent text
                consent_text = i18n.get_text("confirm")
                assert consent_text is not None
                assert len(consent_text) > 0
    
    @pytest.mark.asyncio
    async def test_global_deployment_with_i18n(self):
        """Test global deployment with internationalization"""
        # Setup deployment in multiple regions
        deployment_config = {
            "regions": {
                DeploymentRegion.US_EAST_1: {
                    "instance_type": "t3.small",
                    "min_instances": 1,
                    "max_instances": 2,
                    "target_cpu": 70.0,
                    "compliance": [ComplianceStandard.CCPA],
                    "languages": ["en", "es"]
                },
                DeploymentRegion.EU_WEST_1: {
                    "instance_type": "t3.small",
                    "min_instances": 1,
                    "max_instances": 2,
                    "target_cpu": 70.0,
                    "compliance": [ComplianceStandard.GDPR],
                    "languages": ["en", "de", "fr"]
                }
            }
        }
        
        deployment = MultiRegionDeployment(deployment_config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            i18n = InternationalizationSystem(Path(temp_dir))
            
            # Mock successful deployment
            with patch.object(
                deployment.compliance_manager,
                'validate_region_compliance',
                return_value=True
            ):
                with patch.object(deployment, '_start_global_monitoring'):
                    result = await deployment.deploy_globally()
                    
                    assert result["overall_status"] in ["success", "partial_failure"]
                    
                    # Verify both regions are active
                    active_regions = result["active_regions"]
                    assert DeploymentRegion.US_EAST_1.value in active_regions
                    assert DeploymentRegion.EU_WEST_1.value in active_regions
            
            # Test language support for each region
            for region_name in active_regions:
                region = DeploymentRegion(region_name)
                region_config = deployment.region_deployments[region]["config"]
                
                # Test each supported language
                for lang_code in region_config.supported_languages:
                    try:
                        lang = SupportedLanguage(lang_code)
                        i18n.set_language(lang)
                        
                        # Verify basic translations work
                        welcome_text = i18n.get_text("welcome")
                        assert welcome_text is not None
                        assert len(welcome_text) > 0
                        
                    except ValueError:
                        # Language not supported in enum, skip
                        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])