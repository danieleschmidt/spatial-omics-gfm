"""
Simple internationalization (i18n) system for global deployment.

Provides basic multi-language support without heavy dependencies.
Supports English, Spanish, French, German, Japanese, and Chinese.
"""

import json
import os
from typing import Dict, Optional, Any
from pathlib import Path
import warnings


class SimpleI18nManager:
    """Lightweight internationalization manager."""
    
    def __init__(self, default_locale: str = "en"):
        self.default_locale = default_locale
        self.current_locale = default_locale
        self.translations = {}
        self.fallback_translations = {}
        
        # Load default translations
        self._load_built_in_translations()
    
    def _load_built_in_translations(self):
        """Load built-in translations for core messages."""
        
        # English (default)
        self.fallback_translations = {
            "analysis.start": "Starting spatial transcriptomics analysis...",
            "analysis.complete": "Analysis complete! Results saved to {filename}",
            "analysis.cells": "Cells",
            "analysis.genes": "Genes", 
            "analysis.mean_expression": "Mean expression per cell",
            "analysis.spatial_extent": "Spatial extent",
            "analysis.predicting_cell_types": "Predicting cell types...",
            "analysis.predicting_interactions": "Predicting cell-cell interactions...",
            "analysis.found_interactions": "Found {count} potential interactions",
            "analysis.top_interactions": "Top interactions",
            "analysis.saving_results": "Saving analysis results...",
            "error.invalid_data": "Invalid data provided",
            "error.file_not_found": "File not found: {filename}",
            "error.memory_error": "Insufficient memory for operation",
            "error.dimension_mismatch": "Data dimension mismatch: {details}",
            "warning.large_dataset": "Large dataset detected, processing may take time",
            "warning.no_torch": "PyTorch not available, using simplified mode",
            "success.data_created": "Demo data created successfully",
            "success.validation_passed": "Data validation passed",
            "success.optimization_complete": "Performance optimization complete"
        }
        
        # Multi-language translations
        translations = {
            "es": {  # Spanish
                "analysis.start": "Iniciando análisis de transcriptómica espacial...",
                "analysis.complete": "¡Análisis completo! Resultados guardados en {filename}",
                "analysis.cells": "Células",
                "analysis.genes": "Genes",
                "analysis.mean_expression": "Expresión media por célula",
                "analysis.spatial_extent": "Extensión espacial",
                "analysis.predicting_cell_types": "Prediciendo tipos celulares...",
                "analysis.predicting_interactions": "Prediciendo interacciones célula-célula...",
                "analysis.found_interactions": "Se encontraron {count} interacciones potenciales",
                "analysis.top_interactions": "Principales interacciones",
                "analysis.saving_results": "Guardando resultados del análisis...",
                "error.invalid_data": "Datos inválidos proporcionados",
                "error.file_not_found": "Archivo no encontrado: {filename}",
                "error.memory_error": "Memoria insuficiente para la operación",
                "error.dimension_mismatch": "Desajuste de dimensiones de datos: {details}",
                "warning.large_dataset": "Dataset grande detectado, el procesamiento puede tomar tiempo",
                "warning.no_torch": "PyTorch no disponible, usando modo simplificado",
                "success.data_created": "Datos de demostración creados exitosamente",
                "success.validation_passed": "Validación de datos aprobada",
                "success.optimization_complete": "Optimización de rendimiento completa"
            },
            "fr": {  # French
                "analysis.start": "Démarrage de l'analyse de transcriptomique spatiale...",
                "analysis.complete": "Analyse terminée ! Résultats sauvegardés dans {filename}",
                "analysis.cells": "Cellules",
                "analysis.genes": "Gènes",
                "analysis.mean_expression": "Expression moyenne par cellule",
                "analysis.spatial_extent": "Étendue spatiale",
                "analysis.predicting_cell_types": "Prédiction des types cellulaires...",
                "analysis.predicting_interactions": "Prédiction des interactions cellule-cellule...",
                "analysis.found_interactions": "{count} interactions potentielles trouvées",
                "analysis.top_interactions": "Principales interactions",
                "analysis.saving_results": "Sauvegarde des résultats d'analyse...",
                "error.invalid_data": "Données invalides fournies",
                "error.file_not_found": "Fichier non trouvé : {filename}",
                "error.memory_error": "Mémoire insuffisante pour l'opération",
                "error.dimension_mismatch": "Incompatibilité des dimensions de données : {details}",
                "warning.large_dataset": "Grand dataset détecté, le traitement peut prendre du temps",
                "warning.no_torch": "PyTorch non disponible, utilisation du mode simplifié",
                "success.data_created": "Données de démonstration créées avec succès",
                "success.validation_passed": "Validation des données réussie",
                "success.optimization_complete": "Optimisation des performances terminée"
            },
            "de": {  # German
                "analysis.start": "Starte räumliche Transkriptomanalyse...",
                "analysis.complete": "Analyse abgeschlossen! Ergebnisse gespeichert in {filename}",
                "analysis.cells": "Zellen",
                "analysis.genes": "Gene",
                "analysis.mean_expression": "Mittlere Expression pro Zelle",
                "analysis.spatial_extent": "Räumliche Ausdehnung",
                "analysis.predicting_cell_types": "Vorhersage von Zelltypen...",
                "analysis.predicting_interactions": "Vorhersage von Zell-Zell-Interaktionen...",
                "analysis.found_interactions": "{count} potentielle Interaktionen gefunden",
                "analysis.top_interactions": "Top-Interaktionen",
                "analysis.saving_results": "Speichere Analyseergebnisse...",
                "error.invalid_data": "Ungültige Daten bereitgestellt",
                "error.file_not_found": "Datei nicht gefunden: {filename}",
                "error.memory_error": "Unzureichender Speicher für Operation",
                "error.dimension_mismatch": "Datenmaße stimmen nicht überein: {details}",
                "warning.large_dataset": "Großer Datensatz erkannt, Verarbeitung kann Zeit benötigen",
                "warning.no_torch": "PyTorch nicht verfügbar, verwende vereinfachten Modus",
                "success.data_created": "Demo-Daten erfolgreich erstellt",
                "success.validation_passed": "Datenvalidierung bestanden",
                "success.optimization_complete": "Leistungsoptimierung abgeschlossen"
            },
            "ja": {  # Japanese
                "analysis.start": "空間トランスクリプトーム解析を開始しています...",
                "analysis.complete": "解析完了！結果は{filename}に保存されました",
                "analysis.cells": "細胞",
                "analysis.genes": "遺伝子",
                "analysis.mean_expression": "細胞あたりの平均発現量",
                "analysis.spatial_extent": "空間的範囲",
                "analysis.predicting_cell_types": "細胞タイプを予測しています...",
                "analysis.predicting_interactions": "細胞間相互作用を予測しています...",
                "analysis.found_interactions": "{count}個の潜在的相互作用が見つかりました",
                "analysis.top_interactions": "主要な相互作用",
                "analysis.saving_results": "解析結果を保存しています...",
                "error.invalid_data": "無効なデータが提供されました",
                "error.file_not_found": "ファイルが見つかりません：{filename}",
                "error.memory_error": "操作に必要なメモリが不足しています",
                "error.dimension_mismatch": "データの次元が一致しません：{details}",
                "warning.large_dataset": "大きなデータセットが検出されました。処理に時間がかかる場合があります",
                "warning.no_torch": "PyTorchが利用できません。簡素化モードを使用します",
                "success.data_created": "デモデータが正常に作成されました",
                "success.validation_passed": "データ検証が通過しました",
                "success.optimization_complete": "パフォーマンス最適化が完了しました"
            },
            "zh": {  # Chinese (Simplified)
                "analysis.start": "开始空间转录组学分析...",
                "analysis.complete": "分析完成！结果已保存到{filename}",
                "analysis.cells": "细胞",
                "analysis.genes": "基因",
                "analysis.mean_expression": "每个细胞的平均表达量",
                "analysis.spatial_extent": "空间范围",
                "analysis.predicting_cell_types": "预测细胞类型...",
                "analysis.predicting_interactions": "预测细胞间相互作用...",
                "analysis.found_interactions": "发现{count}个潜在相互作用",
                "analysis.top_interactions": "主要相互作用",
                "analysis.saving_results": "保存分析结果...",
                "error.invalid_data": "提供的数据无效",
                "error.file_not_found": "文件未找到：{filename}",
                "error.memory_error": "操作内存不足",
                "error.dimension_mismatch": "数据维度不匹配：{details}",
                "warning.large_dataset": "检测到大数据集，处理可能需要时间",
                "warning.no_torch": "PyTorch不可用，使用简化模式",
                "success.data_created": "演示数据创建成功",
                "success.validation_passed": "数据验证通过",
                "success.optimization_complete": "性能优化完成"
            }
        }
        
        self.translations = translations
    
    def set_locale(self, locale: str) -> bool:
        """
        Set the current locale.
        
        Args:
            locale: Language code (e.g., 'en', 'es', 'fr', 'de', 'ja', 'zh')
            
        Returns:
            True if locale was set successfully, False otherwise
        """
        if locale in self.translations or locale == self.default_locale:
            self.current_locale = locale
            return True
        else:
            warnings.warn(f"Locale '{locale}' not supported, falling back to '{self.default_locale}'")
            self.current_locale = self.default_locale
            return False
    
    def get_locale(self) -> str:
        """Get the current locale."""
        return self.current_locale
    
    def translate(self, key: str, **kwargs) -> str:
        """
        Translate a key to the current locale.
        
        Args:
            key: Translation key (e.g., 'analysis.start')
            **kwargs: Variables to substitute in the translation
            
        Returns:
            Translated string
        """
        # Try current locale first
        if self.current_locale in self.translations:
            translation = self.translations[self.current_locale].get(key)
            if translation:
                try:
                    return translation.format(**kwargs)
                except (KeyError, ValueError):
                    # Fall back to unformatted string if formatting fails
                    return translation
        
        # Fall back to default/English
        fallback = self.fallback_translations.get(key, key)
        try:
            return fallback.format(**kwargs)
        except (KeyError, ValueError):
            return fallback
    
    def get_supported_locales(self) -> list:
        """Get list of supported locale codes."""
        return [self.default_locale] + list(self.translations.keys())
    
    def load_custom_translations(self, locale: str, translations: Dict[str, str]):
        """
        Load custom translations for a locale.
        
        Args:
            locale: Language code
            translations: Dictionary of key-value translation pairs
        """
        if locale not in self.translations:
            self.translations[locale] = {}
        self.translations[locale].update(translations)


# Alias for compatibility
SimpleI18n = SimpleI18nManager

# Global i18n manager instance
_i18n_manager = SimpleI18nManager()


def set_locale(locale: str) -> bool:
    """Set the global locale."""
    return _i18n_manager.set_locale(locale)


def get_locale() -> str:
    """Get the current global locale."""
    return _i18n_manager.get_locale()


def translate(key: str, **kwargs) -> str:
    """Translate a key using the global i18n manager."""
    return _i18n_manager.translate(key, **kwargs)


def t(key: str, **kwargs) -> str:
    """Short alias for translate()."""
    return translate(key, **kwargs)


def get_supported_locales() -> list:
    """Get list of supported locale codes."""
    return _i18n_manager.get_supported_locales()


def load_custom_translations(locale: str, translations: Dict[str, str]):
    """Load custom translations for a locale."""
    _i18n_manager.load_custom_translations(locale, translations)


class SimpleComplianceChecker:
    """Basic compliance checker for global regulations."""
    
    def __init__(self):
        self.compliance_rules = {
            "GDPR": {
                "description": "General Data Protection Regulation (EU)",
                "requirements": [
                    "Data minimization",
                    "Purpose limitation", 
                    "Storage limitation",
                    "User consent for processing",
                    "Right to deletion"
                ],
                "applicable_regions": ["EU", "EEA"]
            },
            "CCPA": {
                "description": "California Consumer Privacy Act (US)",
                "requirements": [
                    "Right to know about data collection",
                    "Right to delete personal information",
                    "Right to opt-out of sale",
                    "Non-discrimination for exercising rights"
                ],
                "applicable_regions": ["CA", "US"]
            },
            "PDPA": {
                "description": "Personal Data Protection Act (Singapore)",
                "requirements": [
                    "Consent for collection and use",
                    "Purpose limitation",
                    "Data protection obligations",
                    "Individual access rights"
                ],
                "applicable_regions": ["SG"]
            }
        }
    
    def check_compliance(self, region: str) -> Dict[str, Any]:
        """
        Check compliance requirements for a region.
        
        Args:
            region: Region code (e.g., 'EU', 'US', 'CA', 'SG')
            
        Returns:
            Dictionary with compliance information
        """
        applicable_rules = []
        
        for rule_name, rule_info in self.compliance_rules.items():
            if region in rule_info["applicable_regions"]:
                applicable_rules.append({
                    "name": rule_name,
                    "description": rule_info["description"],
                    "requirements": rule_info["requirements"]
                })
        
        return {
            "region": region,
            "applicable_rules": applicable_rules,
            "compliance_status": "review_required" if applicable_rules else "no_specific_rules",
            "recommendations": self._get_recommendations(region, applicable_rules)
        }
    
    def _get_recommendations(self, region: str, rules: list) -> list:
        """Get compliance recommendations for a region."""
        if not rules:
            return ["No specific data protection rules identified for this region"]
        
        recommendations = []
        for rule in rules:
            recommendations.extend([
                f"Ensure compliance with {rule['name']}: {rule['description']}",
                f"Implement requirements: {', '.join(rule['requirements'])}"
            ])
        
        # Add general recommendations
        recommendations.extend([
            "Implement data encryption at rest and in transit",
            "Maintain audit logs for data access and processing",
            "Establish data retention and deletion policies",
            "Provide clear privacy policy and user consent mechanisms"
        ])
        
        return recommendations


def test_i18n_system():
    """Test the internationalization system."""
    print("=== Testing Simple I18n System ===")
    
    # Test supported locales
    locales = get_supported_locales()
    print(f"Supported locales: {locales}")
    
    # Test translations in different languages
    test_key = "analysis.start"
    
    for locale in locales:
        set_locale(locale)
        translation = translate(test_key)
        print(f"{locale}: {translation}")
    
    # Test with parameters
    set_locale("en")
    message = translate("analysis.complete", filename="test_results.json")
    print(f"English with params: {message}")
    
    set_locale("es")
    message = translate("analysis.found_interactions", count=150)
    print(f"Spanish with params: {message}")
    
    # Test compliance checker
    print("\n=== Testing Compliance Checker ===")
    checker = SimpleComplianceChecker()
    
    for region in ["EU", "US", "SG", "JP"]:
        compliance = checker.check_compliance(region)
        print(f"\n{region} Compliance:")
        print(f"  Status: {compliance['compliance_status']}")
        print(f"  Rules: {len(compliance['applicable_rules'])}")
        if compliance['applicable_rules']:
            for rule in compliance['applicable_rules']:
                print(f"    - {rule['name']}: {rule['description']}")
    
    print("\n✅ I18n and compliance system working")


if __name__ == "__main__":
    test_i18n_system()