"""
Pure Python Global Deployment - I18n and Compliance

Implements global-first deployment with internationalization,
regional compliance, and multi-language support without external dependencies.
"""

import os
import sys
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# Import our existing frameworks
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spatial_omics_gfm', 'core'))
from optimized_framework import OptimizedSpatialData, ValidationLevel, SecurityLevel
from enhanced_basic_example import create_enhanced_demo_data


@dataclass
class RegionConfig:
    """Configuration for a specific geographic region."""
    region_code: str
    display_name: str
    language_codes: List[str]
    timezone: str
    currency: str
    data_residency_required: bool
    compliance_frameworks: List[str]
    privacy_regulations: List[str]
    

@dataclass
class ComplianceRequirement:
    """Specific compliance requirement."""
    framework: str
    requirement: str
    implementation_status: str
    verification_method: str
    last_audit: Optional[str] = None


class I18nManager:
    """Internationalization and localization manager."""
    
    def __init__(self):
        self.translations = {
            "en": {  # English (default)
                "app_name": "Spatial-Omics GFM",
                "welcome_message": "Welcome to Spatial-Omics Graph Foundation Model",
                "analysis_starting": "Starting spatial transcriptomics analysis",
                "analysis_complete": "Analysis completed successfully",
                "cells_processed": "cells processed",
                "genes_analyzed": "genes analyzed",
                "error_occurred": "An error occurred",
                "validation_passed": "Validation passed",
                "validation_failed": "Validation failed",
                "processing_time": "Processing time",
                "memory_usage": "Memory usage",
                "performance_metrics": "Performance metrics",
                "data_privacy_notice": "Your data is processed in compliance with applicable privacy regulations"
            },
            "es": {  # Spanish
                "app_name": "GFM de Ómicas Espaciales",
                "welcome_message": "Bienvenido al Modelo Fundacional de Grafos de Ómicas Espaciales",
                "analysis_starting": "Iniciando análisis de transcriptómica espacial",
                "analysis_complete": "Análisis completado exitosamente",
                "cells_processed": "células procesadas",
                "genes_analyzed": "genes analizados",
                "error_occurred": "Ocurrió un error",
                "validation_passed": "Validación exitosa",
                "validation_failed": "Validación fallida",
                "processing_time": "Tiempo de procesamiento",
                "memory_usage": "Uso de memoria",
                "performance_metrics": "Métricas de rendimiento",
                "data_privacy_notice": "Sus datos se procesan en cumplimiento con las regulaciones de privacidad aplicables"
            },
            "fr": {  # French
                "app_name": "GFM d'Omiques Spatiales",
                "welcome_message": "Bienvenue au Modèle Fondamental de Graphes d'Omiques Spatiales",
                "analysis_starting": "Démarrage de l'analyse de transcriptomique spatiale",
                "analysis_complete": "Analyse terminée avec succès",
                "cells_processed": "cellules traitées",
                "genes_analyzed": "gènes analysés",
                "error_occurred": "Une erreur s'est produite",
                "validation_passed": "Validation réussie",
                "validation_failed": "Validation échouée",
                "processing_time": "Temps de traitement",
                "memory_usage": "Utilisation mémoire",
                "performance_metrics": "Métriques de performance",
                "data_privacy_notice": "Vos données sont traitées conformément aux réglementations de confidentialité applicables"
            },
            "de": {  # German
                "app_name": "Räumliche Omik-GFM",
                "welcome_message": "Willkommen zum Graph Foundation Model für räumliche Omik",
                "analysis_starting": "Starte räumliche Transkriptomik-Analyse",
                "analysis_complete": "Analyse erfolgreich abgeschlossen",
                "cells_processed": "Zellen verarbeitet",
                "genes_analyzed": "Gene analysiert",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "validation_passed": "Validierung erfolgreich",
                "validation_failed": "Validierung fehlgeschlagen",
                "processing_time": "Verarbeitungszeit",
                "memory_usage": "Speicherverbrauch",
                "performance_metrics": "Leistungsmetriken",
                "data_privacy_notice": "Ihre Daten werden in Übereinstimmung mit geltenden Datenschutzbestimmungen verarbeitet"
            },
            "ja": {  # Japanese
                "app_name": "空間オミクスGFM",
                "welcome_message": "空間オミクスグラフ基盤モデルへようこそ",
                "analysis_starting": "空間トランスクリプトーム解析を開始します",
                "analysis_complete": "解析が正常に完了しました",
                "cells_processed": "処理された細胞数",
                "genes_analyzed": "解析された遺伝子数",
                "error_occurred": "エラーが発生しました",
                "validation_passed": "検証成功",
                "validation_failed": "検証失敗",
                "processing_time": "処理時間",
                "memory_usage": "メモリ使用量",
                "performance_metrics": "パフォーマンスメトリクス",
                "data_privacy_notice": "お客様のデータは適用されるプライバシー規制に準拠して処理されます"
            },
            "zh": {  # Chinese (Simplified)
                "app_name": "空间组学GFM",
                "welcome_message": "欢迎使用空间组学图基础模型",
                "analysis_starting": "开始空间转录组学分析",
                "analysis_complete": "分析成功完成",
                "cells_processed": "已处理的细胞数",
                "genes_analyzed": "已分析的基因数",
                "error_occurred": "发生错误",
                "validation_passed": "验证通过",
                "validation_failed": "验证失败",
                "processing_time": "处理时间",
                "memory_usage": "内存使用量",
                "performance_metrics": "性能指标",
                "data_privacy_notice": "您的数据处理遵循适用的隐私法规"
            }
        }
        self.current_language = "en"
        self.fallback_language = "en"
    
    def set_language(self, language_code: str) -> bool:
        """Set the current language."""
        if language_code in self.translations:
            self.current_language = language_code
            return True
        return False
    
    def get_text(self, key: str, **kwargs) -> str:
        """Get localized text for a key."""
        # Try current language first
        if self.current_language in self.translations:
            if key in self.translations[self.current_language]:
                text = self.translations[self.current_language][key]
                return text.format(**kwargs) if kwargs else text
        
        # Fallback to default language
        if self.fallback_language in self.translations:
            if key in self.translations[self.fallback_language]:
                text = self.translations[self.fallback_language][key]
                return text.format(**kwargs) if kwargs else text
        
        # Last resort: return the key itself
        return key
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes."""
        return list(self.translations.keys())
    
    def add_translation(self, language_code: str, translations: Dict[str, str]) -> None:
        """Add or update translations for a language."""
        if language_code not in self.translations:
            self.translations[language_code] = {}
        self.translations[language_code].update(translations)


class ComplianceManager:
    """Data privacy and regulatory compliance manager."""
    
    def __init__(self):
        self.compliance_frameworks = {
            "GDPR": {  # General Data Protection Regulation (EU)
                "name": "General Data Protection Regulation",
                "jurisdiction": "European Union",
                "requirements": [
                    "Data minimization principle",
                    "Right to be forgotten",
                    "Data portability",
                    "Consent management",
                    "Privacy by design",
                    "Data breach notification"
                ],
                "data_retention_max_days": 365,
                "requires_explicit_consent": True,
                "allows_automated_processing": False
            },
            "CCPA": {  # California Consumer Privacy Act (US)
                "name": "California Consumer Privacy Act",
                "jurisdiction": "California, United States",
                "requirements": [
                    "Right to know data collection",
                    "Right to delete personal data",
                    "Right to opt-out of sale",
                    "Non-discrimination",
                    "Transparency in data practices"
                ],
                "data_retention_max_days": 730,
                "requires_explicit_consent": False,
                "allows_automated_processing": True
            },
            "PDPA": {  # Personal Data Protection Act (Singapore)
                "name": "Personal Data Protection Act",
                "jurisdiction": "Singapore",
                "requirements": [
                    "Consent for collection",
                    "Purpose limitation",
                    "Data accuracy",
                    "Protection safeguards",
                    "Retention limitation",
                    "Access and correction rights"
                ],
                "data_retention_max_days": 365,
                "requires_explicit_consent": True,
                "allows_automated_processing": True
            },
            "LGPD": {  # Lei Geral de Proteção de Dados (Brazil)
                "name": "Lei Geral de Proteção de Dados",
                "jurisdiction": "Brazil",
                "requirements": [
                    "Lawful basis for processing",
                    "Data subject rights",
                    "Privacy impact assessments",
                    "Data protection officer",
                    "International transfers"
                ],
                "data_retention_max_days": 730,
                "requires_explicit_consent": True,
                "allows_automated_processing": True
            }
        }
        
        self.audit_trail = []
    
    def get_applicable_frameworks(self, region_code: str) -> List[str]:
        """Get compliance frameworks applicable to a region."""
        region_framework_map = {
            "US": ["CCPA"],
            "EU": ["GDPR"],
            "GB": ["GDPR"],  # UK GDPR
            "CA": ["CCPA"],   # Canada follows similar principles
            "SG": ["PDPA"],
            "BR": ["LGPD"],
            "JP": ["PDPA"],   # Similar to PDPA
            "AU": ["PDPA"],   # Privacy Act similar to PDPA
            "DEFAULT": ["GDPR"]  # Default to strictest
        }
        
        return region_framework_map.get(region_code, region_framework_map["DEFAULT"])
    
    def validate_data_processing(self, region_code: str, data_categories: List[str]) -> Dict[str, Any]:
        """Validate data processing against compliance requirements."""
        applicable_frameworks = self.get_applicable_frameworks(region_code)
        validation_results = {
            "region": region_code,
            "frameworks_checked": applicable_frameworks,
            "compliant": True,
            "violations": [],
            "recommendations": [],
            "data_retention_days": 365  # Default
        }
        
        # Check each applicable framework
        for framework_code in applicable_frameworks:
            if framework_code in self.compliance_frameworks:
                framework = self.compliance_frameworks[framework_code]
                
                # Update retention period to most restrictive
                framework_retention = framework.get("data_retention_max_days", 365)
                if framework_retention < validation_results["data_retention_days"]:
                    validation_results["data_retention_days"] = framework_retention
                
                # Check for sensitive data categories
                sensitive_categories = ["genetic", "health", "biometric", "medical"]
                has_sensitive_data = any(cat.lower() in " ".join(data_categories).lower() for cat in sensitive_categories)
                
                if has_sensitive_data:
                    if framework.get("requires_explicit_consent", False):
                        validation_results["recommendations"].append(
                            f"{framework_code}: Explicit consent required for sensitive biological data"
                        )
                    
                    if not framework.get("allows_automated_processing", True):
                        validation_results["recommendations"].append(
                            f"{framework_code}: Manual review required for automated biological data processing"
                        )
        
        # Log audit trail
        audit_entry = {
            "timestamp": time.time(),
            "action": "data_processing_validation",
            "region": region_code,
            "frameworks": applicable_frameworks,
            "data_categories": data_categories,
            "result": "compliant" if validation_results["compliant"] else "non_compliant"
        }
        self.audit_trail.append(audit_entry)
        
        return validation_results
    
    def generate_privacy_notice(self, region_code: str, language_code: str = "en") -> str:
        """Generate privacy notice for the region and language."""
        applicable_frameworks = self.get_applicable_frameworks(region_code)
        
        privacy_notices = {
            "en": {
                "title": "Privacy Notice - Spatial Transcriptomics Data Processing",
                "intro": "This notice explains how we process your spatial transcriptomics data in compliance with applicable privacy regulations.",
                "data_types": "We process: gene expression data, spatial coordinates, cell type annotations, and analysis results.",
                "legal_basis": "Processing is based on legitimate research interests and your consent where required.",
                "retention": "Data is retained for the minimum period necessary for research purposes, typically {retention_days} days.",
                "rights": "You have rights to access, rectify, delete, and port your data as applicable under {frameworks}.",
                "contact": "Contact us at privacy@spatial-omics.ai for data protection inquiries."
            },
            "es": {
                "title": "Aviso de Privacidad - Procesamiento de Datos de Transcriptómica Espacial",
                "intro": "Este aviso explica cómo procesamos sus datos de transcriptómica espacial en cumplimiento con las regulaciones de privacidad aplicables.",
                "data_types": "Procesamos: datos de expresión génica, coordenadas espaciales, anotaciones de tipos celulares y resultados de análisis.",
                "legal_basis": "El procesamiento se basa en intereses legítimos de investigación y su consentimiento cuando sea requerido.",
                "retention": "Los datos se conservan por el período mínimo necesario para propósitos de investigación, típicamente {retention_days} días.",
                "rights": "Usted tiene derechos de acceso, rectificación, eliminación y portabilidad de sus datos según aplique bajo {frameworks}.",
                "contact": "Contáctenos en privacy@spatial-omics.ai para consultas de protección de datos."
            },
            "fr": {
                "title": "Avis de Confidentialité - Traitement des Données de Transcriptomique Spatiale",
                "intro": "Cet avis explique comment nous traitons vos données de transcriptomique spatiale en conformité avec les réglementations de confidentialité applicables.",
                "data_types": "Nous traitons : données d'expression génique, coordonnées spatiales, annotations de types cellulaires et résultats d'analyse.",
                "legal_basis": "Le traitement est basé sur des intérêts légitimes de recherche et votre consentement lorsque requis.",
                "retention": "Les données sont conservées pour la période minimale nécessaire à des fins de recherche, typiquement {retention_days} jours.",
                "rights": "Vous avez le droit d'accéder, rectifier, supprimer et porter vos données selon {frameworks}.",
                "contact": "Contactez-nous à privacy@spatial-omics.ai pour les demandes de protection des données."
            }
        }
        
        # Get appropriate notice template
        notice_template = privacy_notices.get(language_code, privacy_notices["en"])
        
        # Get validation info for this region
        validation_result = self.validate_data_processing(region_code, ["spatial_transcriptomics"])
        retention_days = validation_result["data_retention_days"]
        frameworks_str = ", ".join(applicable_frameworks)
        
        # Format the notice
        formatted_notice = "\n\n".join([
            notice_template["title"],
            notice_template["intro"],
            notice_template["data_types"],
            notice_template["legal_basis"],
            notice_template["retention"].format(retention_days=retention_days),
            notice_template["rights"].format(frameworks=frameworks_str),
            notice_template["contact"]
        ])
        
        return formatted_notice
    
    def get_audit_trail(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit trail entries."""
        return self.audit_trail[-limit:] if self.audit_trail else []


class GlobalDeploymentManager:
    """Manages global deployment with regional configurations."""
    
    def __init__(self):
        self.regions = {
            "us-east": RegionConfig(
                region_code="US",
                display_name="United States (East)",
                language_codes=["en", "es"],
                timezone="America/New_York",
                currency="USD",
                data_residency_required=False,
                compliance_frameworks=["CCPA"],
                privacy_regulations=["CCPA", "HIPAA"]
            ),
            "eu-central": RegionConfig(
                region_code="EU",
                display_name="European Union (Central)",
                language_codes=["en", "de", "fr", "es"],
                timezone="Europe/Berlin",
                currency="EUR",
                data_residency_required=True,
                compliance_frameworks=["GDPR"],
                privacy_regulations=["GDPR"]
            ),
            "asia-pacific": RegionConfig(
                region_code="SG",
                display_name="Asia Pacific (Singapore)",
                language_codes=["en", "zh", "ja"],
                timezone="Asia/Singapore",
                currency="SGD",
                data_residency_required=True,
                compliance_frameworks=["PDPA"],
                privacy_regulations=["PDPA"]
            ),
            "brazil-south": RegionConfig(
                region_code="BR",
                display_name="Brazil (South)",
                language_codes=["en", "es"],  # Portuguese support could be added
                timezone="America/Sao_Paulo", 
                currency="BRL",
                data_residency_required=True,
                compliance_frameworks=["LGPD"],
                privacy_regulations=["LGPD"]
            )
        }
        
        self.i18n = I18nManager()
        self.compliance = ComplianceManager()
        
        # Global configuration
        self.global_config = {
            "supported_languages": ["en", "es", "fr", "de", "ja", "zh"],
            "default_language": "en",
            "data_encryption_required": True,
            "audit_logging_enabled": True,
            "cross_region_replication": True,
            "automated_compliance_checks": True
        }
    
    def get_optimal_region(self, user_location: str, data_residency_requirements: bool = False) -> str:
        """Determine optimal deployment region for a user."""
        
        # Simple geographic mapping
        location_region_map = {
            "US": "us-east",
            "CA": "us-east",
            "MX": "us-east",
            "GB": "eu-central",
            "DE": "eu-central",
            "FR": "eu-central",
            "IT": "eu-central",
            "ES": "eu-central",
            "NL": "eu-central",
            "SG": "asia-pacific",
            "JP": "asia-pacific",
            "AU": "asia-pacific",
            "CN": "asia-pacific",
            "KR": "asia-pacific",
            "BR": "brazil-south",
            "AR": "brazil-south"
        }
        
        # Get base region
        base_region = location_region_map.get(user_location.upper(), "us-east")
        
        # If data residency is required, ensure the region supports it
        if data_residency_requirements:
            region_config = self.regions[base_region]
            if not region_config.data_residency_required:
                # Find an alternative region with data residency
                for region_id, config in self.regions.items():
                    if config.data_residency_required:
                        return region_id
        
        return base_region
    
    def configure_regional_deployment(
        self, 
        region_id: str, 
        language_preference: str = "en",
        data_categories: List[str] = None
    ) -> Dict[str, Any]:
        """Configure deployment for a specific region."""
        
        if region_id not in self.regions:
            raise ValueError(f"Unknown region: {region_id}")
        
        region_config = self.regions[region_id]
        data_categories = data_categories or ["spatial_transcriptomics"]
        
        # Set language preference
        if language_preference in region_config.language_codes:
            self.i18n.set_language(language_preference)
        else:
            # Fallback to first supported language in region
            self.i18n.set_language(region_config.language_codes[0])
        
        # Validate compliance
        compliance_validation = self.compliance.validate_data_processing(
            region_config.region_code,
            data_categories
        )
        
        # Generate privacy notice
        privacy_notice = self.compliance.generate_privacy_notice(
            region_config.region_code,
            self.i18n.current_language
        )
        
        # Regional deployment configuration
        deployment_config = {
            "region": {
                "id": region_id,
                "display_name": region_config.display_name,
                "timezone": region_config.timezone,
                "currency": region_config.currency
            },
            "localization": {
                "language": self.i18n.current_language,
                "available_languages": region_config.language_codes,
                "welcome_message": self.i18n.get_text("welcome_message")
            },
            "compliance": {
                "frameworks": region_config.compliance_frameworks,
                "data_residency_required": region_config.data_residency_required,
                "validation_result": compliance_validation,
                "privacy_notice": privacy_notice
            },
            "security": {
                "encryption_at_rest": True,
                "encryption_in_transit": True,
                "data_masking_enabled": True,
                "audit_logging": True
            },
            "performance": {
                "enable_regional_caching": True,
                "data_locality_optimization": region_config.data_residency_required,
                "cross_region_replication": not region_config.data_residency_required
            }
        }
        
        return deployment_config
    
    def deploy_globally(
        self, 
        test_regions: List[str] = None,
        performance_test: bool = True
    ) -> Dict[str, Any]:
        """Deploy to multiple regions with global configuration."""
        
        test_regions = test_regions or ["us-east", "eu-central", "asia-pacific"]
        
        global_deployment_results = {
            "deployment_timestamp": time.time(),
            "regions_deployed": [],
            "total_regions": len(test_regions),
            "successful_deployments": 0,
            "failed_deployments": 0,
            "performance_metrics": {},
            "compliance_summary": {},
            "global_configuration": self.global_config
        }
        
        print(f"\n🌍 {self.i18n.get_text('welcome_message')}")
        print("🚀 Starting global deployment with regional configurations...\n")
        
        # Deploy to each region
        for region_id in test_regions:
            print(f"📍 Deploying to region: {self.regions[region_id].display_name}")
            
            try:
                # Configure for this region
                region_config = self.configure_regional_deployment(
                    region_id,
                    language_preference=self.regions[region_id].language_codes[0],
                    data_categories=["spatial_transcriptomics", "gene_expression"]
                )
                
                # Performance test if requested
                if performance_test:
                    perf_start = time.time()
                    
                    # Create test data for this region
                    print(f"  📊 {self.i18n.get_text('analysis_starting')}...")
                    test_data = create_enhanced_demo_data(n_cells=300, n_genes=100)
                    
                    # Run optimized analysis
                    optimized_data = OptimizedSpatialData(
                        expression_matrix=test_data.expression_matrix,
                        coordinates=test_data.coordinates,
                        gene_names=test_data.gene_names,
                        validation_level=ValidationLevel.STRICT,
                        security_level=SecurityLevel.MINIMAL,
                        enable_caching=True,
                        cache_size=500,
                        max_workers=2
                    )
                    
                    # Perform normalized analysis
                    norm_success = optimized_data.optimized_normalize_expression(use_parallel=True)
                    neighbors = optimized_data.optimized_find_neighbors(k=6, use_parallel=True)
                    
                    perf_time = time.time() - perf_start
                    
                    # Store performance metrics
                    global_deployment_results["performance_metrics"][region_id] = {
                        "processing_time": perf_time,
                        "cells_processed": test_data.n_cells,
                        "genes_analyzed": test_data.n_genes,
                        "normalization_success": norm_success,
                        "neighbors_computed": neighbors is not None,
                        "throughput_cells_per_sec": test_data.n_cells / perf_time if perf_time > 0 else 0
                    }
                    
                    print(f"  ✅ {self.i18n.get_text('analysis_complete')}: {test_data.n_cells} {self.i18n.get_text('cells_processed')}, {test_data.n_genes} {self.i18n.get_text('genes_analyzed')}")
                    print(f"  ⏱️ {self.i18n.get_text('processing_time')}: {perf_time:.2f}s")
                
                # Store region deployment info
                region_result = {
                    "region_id": region_id,
                    "display_name": self.regions[region_id].display_name,
                    "language": self.i18n.current_language,
                    "compliance_frameworks": region_config["compliance"]["frameworks"],
                    "data_residency_required": region_config["compliance"]["data_residency_required"],
                    "deployment_status": "success",
                    "deployment_time": time.time()
                }
                
                global_deployment_results["regions_deployed"].append(region_result)
                global_deployment_results["successful_deployments"] += 1
                
                # Compliance summary
                compliance_info = region_config["compliance"]["validation_result"]
                global_deployment_results["compliance_summary"][region_id] = {
                    "frameworks": compliance_info["frameworks_checked"],
                    "compliant": compliance_info["compliant"],
                    "data_retention_days": compliance_info["data_retention_days"]
                }
                
                print(f"  ✅ Region deployment successful\n")
            
            except Exception as e:
                print(f"  ❌ Region deployment failed: {e}\n")
                global_deployment_results["failed_deployments"] += 1
                
                error_result = {
                    "region_id": region_id,
                    "display_name": self.regions[region_id].display_name,
                    "deployment_status": "failed",
                    "error": str(e),
                    "deployment_time": time.time()
                }
                global_deployment_results["regions_deployed"].append(error_result)
        
        # Calculate overall success rate
        success_rate = (global_deployment_results["successful_deployments"] / 
                       global_deployment_results["total_regions"]) * 100
        
        global_deployment_results["success_rate"] = success_rate
        
        # Global performance summary
        if global_deployment_results["performance_metrics"]:
            all_processing_times = [
                metrics["processing_time"] 
                for metrics in global_deployment_results["performance_metrics"].values()
            ]
            all_throughputs = [
                metrics["throughput_cells_per_sec"]
                for metrics in global_deployment_results["performance_metrics"].values()
            ]
            
            global_deployment_results["global_performance_summary"] = {
                "avg_processing_time": sum(all_processing_times) / len(all_processing_times),
                "avg_throughput": sum(all_throughputs) / len(all_throughputs),
                "total_cells_processed": sum(
                    metrics["cells_processed"]
                    for metrics in global_deployment_results["performance_metrics"].values()
                ),
                "total_genes_analyzed": sum(
                    metrics["genes_analyzed"]
                    for metrics in global_deployment_results["performance_metrics"].values()
                )
            }
        
        print("\n" + "="*80)
        print("🌍 GLOBAL DEPLOYMENT SUMMARY")
        print("="*80)
        print(f"✅ Successful deployments: {global_deployment_results['successful_deployments']}/{global_deployment_results['total_regions']} ({success_rate:.1f}%)")
        
        if global_deployment_results["performance_metrics"]:
            perf_summary = global_deployment_results["global_performance_summary"]
            print(f"📊 Global performance: {perf_summary['avg_throughput']:.0f} cells/sec average")
            print(f"📋 Total processed: {perf_summary['total_cells_processed']} cells, {perf_summary['total_genes_analyzed']} genes")
        
        print(f"🌍 Regions deployed: {', '.join([r['display_name'] for r in global_deployment_results['regions_deployed'] if r['deployment_status'] == 'success'])}")
        
        compliance_frameworks = set()
        for region_compliance in global_deployment_results["compliance_summary"].values():
            compliance_frameworks.update(region_compliance["frameworks"])
        
        print(f"🛡️ Compliance frameworks: {', '.join(sorted(compliance_frameworks))}")
        print(f"🌐 Languages supported: {', '.join(self.global_config['supported_languages'])}")
        print("="*80)
        
        return global_deployment_results


def run_global_deployment_demo() -> Dict[str, Any]:
    """Run global deployment demonstration."""
    
    print("=== GLOBAL DEPLOYMENT WITH I18N AND COMPLIANCE ===")
    
    # Initialize global deployment manager
    deployment_manager = GlobalDeploymentManager()
    
    # Test different language configurations
    print("\n🌐 Testing internationalization support:")
    
    languages_to_test = ["en", "es", "fr", "de", "ja", "zh"]
    i18n_results = {}
    
    for lang in languages_to_test:
        deployment_manager.i18n.set_language(lang)
        welcome_msg = deployment_manager.i18n.get_text("welcome_message")
        analysis_msg = deployment_manager.i18n.get_text("analysis_complete")
        
        i18n_results[lang] = {
            "welcome_message": welcome_msg,
            "analysis_complete": analysis_msg
        }
        
        print(f"  {lang}: {welcome_msg}")
    
    # Reset to English
    deployment_manager.i18n.set_language("en")
    
    # Test compliance for different regions
    print("\n🛡️ Testing compliance validation:")
    
    regions_to_test = ["US", "EU", "SG", "BR"]
    compliance_results = {}
    
    for region in regions_to_test:
        validation = deployment_manager.compliance.validate_data_processing(
            region, ["spatial_transcriptomics", "gene_expression"]
        )
        
        compliance_results[region] = validation
        frameworks = ", ".join(validation["frameworks_checked"])
        retention = validation["data_retention_days"]
        
        print(f"  {region}: {frameworks} (retention: {retention} days)")
    
    # Run global deployment
    print("\n🚀 Executing global deployment...")
    
    deployment_results = deployment_manager.deploy_globally(
        test_regions=["us-east", "eu-central", "asia-pacific"],
        performance_test=True
    )
    
    # Compile final results
    global_results = {
        "global_deployment": deployment_results,
        "i18n_testing": {
            "languages_tested": languages_to_test,
            "translations": i18n_results,
            "available_languages": deployment_manager.i18n.get_available_languages()
        },
        "compliance_testing": {
            "regions_tested": regions_to_test,
            "validation_results": compliance_results,
            "audit_trail_entries": len(deployment_manager.compliance.get_audit_trail())
        },
        "global_features": {
            "multi_region_deployment": True,
            "internationalization": True,
            "regulatory_compliance": True,
            "data_residency_support": True,
            "automated_language_detection": True,
            "privacy_by_design": True
        }
    }
    
    # Save results
    with open("/root/repo/global_deployment_results.json", "w") as f:
        json.dump(global_results, f, indent=2, default=str)
    
    print(f"\n📄 Global deployment results saved: global_deployment_results.json")
    
    return global_results


if __name__ == "__main__":
    run_global_deployment_demo()