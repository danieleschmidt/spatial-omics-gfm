"""
Global-First Deployment Module
Multi-region deployment and compliance systems
"""

import warnings

# Simple global features (always available)
try:
    from .simple_i18n import (
        SimpleI18nManager, SimpleComplianceChecker, 
        set_locale, get_locale, translate, t, 
        get_supported_locales, load_custom_translations
    )
    SIMPLE_GLOBAL_FEATURES = True
except ImportError:
    SIMPLE_GLOBAL_FEATURES = False
    SimpleI18nManager = SimpleComplianceChecker = None
    set_locale = get_locale = translate = t = None
    get_supported_locales = load_custom_translations = None
    warnings.warn("Simple global features not available", ImportWarning)

# Advanced global features (optional)
try:
    from .multi_region_deployment import (
        MultiRegionDeployment,
        DeploymentRegion,
        DeploymentStatus,
        ComplianceStandard,
        RegionConfig,
        DeploymentMetrics
    )
    from .i18n_system import InternationalizationSystem
    from .compliance_engine import ComplianceEngine
    ADVANCED_GLOBAL_FEATURES = True
except ImportError:
    ADVANCED_GLOBAL_FEATURES = False
    MultiRegionDeployment = DeploymentRegion = DeploymentStatus = None
    ComplianceStandard = RegionConfig = DeploymentMetrics = None
    InternationalizationSystem = ComplianceEngine = None
    warnings.warn("Advanced global features not available - some dependencies missing", ImportWarning)

__all__ = [
    # Simple global features
    "SimpleI18nManager",
    "SimpleComplianceChecker", 
    "set_locale",
    "get_locale", 
    "translate",
    "t",
    "get_supported_locales",
    "load_custom_translations",
    "SIMPLE_GLOBAL_FEATURES",
    
    # Advanced global features
    "MultiRegionDeployment",
    "DeploymentRegion",
    "DeploymentStatus", 
    "ComplianceStandard",
    "RegionConfig",
    "DeploymentMetrics",
    "InternationalizationSystem",
    "ComplianceEngine",
    "ADVANCED_GLOBAL_FEATURES"
]