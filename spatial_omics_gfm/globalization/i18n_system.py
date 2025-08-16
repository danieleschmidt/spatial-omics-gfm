"""
Global Internationalization System
Multi-language support for global deployment
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class SupportedLanguage(Enum):
    """Supported languages for internationalization"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh"
    KOREAN = "ko"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    DUTCH = "nl"


@dataclass
class LocalizationConfig:
    """Configuration for localization"""
    language: SupportedLanguage
    region: str
    date_format: str
    number_format: str
    currency_format: str
    timezone: str
    rtl_support: bool = False


class InternationalizationSystem:
    """
    Global Internationalization System
    
    Provides comprehensive multi-language support with:
    - Dynamic language switching
    - Regional formatting
    - Cultural adaptations
    - Accessibility compliance
    - Content localization
    """
    
    def __init__(self, base_path: Path, default_language: SupportedLanguage = SupportedLanguage.ENGLISH):
        self.base_path = Path(base_path)
        self.default_language = default_language
        self.current_language = default_language
        self.logger = self._setup_logging()
        
        # Translation data
        self.translations: Dict[SupportedLanguage, Dict[str, str]] = {}
        self.localization_configs: Dict[SupportedLanguage, LocalizationConfig] = {}
        
        # Initialize system
        self._initialize_translations()
        self._initialize_localization_configs()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for i18n system"""
        logger = logging.getLogger("i18n_system")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_translations(self) -> None:
        """Initialize translation dictionaries"""
        # Core system translations
        self.translations = {
            SupportedLanguage.ENGLISH: {
                "welcome": "Welcome to Spatial-Omics GFM",
                "model_loading": "Loading model...",
                "analysis_complete": "Analysis complete",
                "error_occurred": "An error occurred",
                "processing": "Processing...",
                "results": "Results",
                "settings": "Settings",
                "help": "Help",
                "about": "About",
                "language": "Language",
                "performance": "Performance",
                "accuracy": "Accuracy",
                "speed": "Speed",
                "memory_usage": "Memory Usage",
                "loading_data": "Loading data...",
                "saving_results": "Saving results...",
                "export": "Export",
                "import": "Import",
                "cancel": "Cancel",
                "confirm": "Confirm",
                "yes": "Yes",
                "no": "No"
            },
            SupportedLanguage.SPANISH: {
                "welcome": "Bienvenido a Spatial-Omics GFM",
                "model_loading": "Cargando modelo...",
                "analysis_complete": "Análisis completo",
                "error_occurred": "Ha ocurrido un error",
                "processing": "Procesando...",
                "results": "Resultados",
                "settings": "Configuración",
                "help": "Ayuda",
                "about": "Acerca de",
                "language": "Idioma",
                "performance": "Rendimiento",
                "accuracy": "Precisión",
                "speed": "Velocidad",
                "memory_usage": "Uso de Memoria",
                "loading_data": "Cargando datos...",
                "saving_results": "Guardando resultados...",
                "export": "Exportar",
                "import": "Importar",
                "cancel": "Cancelar",
                "confirm": "Confirmar",
                "yes": "Sí",
                "no": "No"
            },
            SupportedLanguage.FRENCH: {
                "welcome": "Bienvenue dans Spatial-Omics GFM",
                "model_loading": "Chargement du modèle...",
                "analysis_complete": "Analyse terminée",
                "error_occurred": "Une erreur s'est produite",
                "processing": "Traitement en cours...",
                "results": "Résultats",
                "settings": "Paramètres",
                "help": "Aide",
                "about": "À propos",
                "language": "Langue",
                "performance": "Performance",
                "accuracy": "Précision",
                "speed": "Vitesse",
                "memory_usage": "Utilisation Mémoire",
                "loading_data": "Chargement des données...",
                "saving_results": "Sauvegarde des résultats...",
                "export": "Exporter",
                "import": "Importer",
                "cancel": "Annuler",
                "confirm": "Confirmer",
                "yes": "Oui",
                "no": "Non"
            },
            SupportedLanguage.GERMAN: {
                "welcome": "Willkommen bei Spatial-Omics GFM",
                "model_loading": "Modell wird geladen...",
                "analysis_complete": "Analyse abgeschlossen",
                "error_occurred": "Ein Fehler ist aufgetreten",
                "processing": "Verarbeitung...",
                "results": "Ergebnisse",
                "settings": "Einstellungen",
                "help": "Hilfe",
                "about": "Über",
                "language": "Sprache",
                "performance": "Leistung",
                "accuracy": "Genauigkeit",
                "speed": "Geschwindigkeit",
                "memory_usage": "Speicherverbrauch",
                "loading_data": "Daten werden geladen...",
                "saving_results": "Ergebnisse werden gespeichert...",
                "export": "Exportieren",
                "import": "Importieren",
                "cancel": "Abbrechen",
                "confirm": "Bestätigen",
                "yes": "Ja",
                "no": "Nein"
            },
            SupportedLanguage.JAPANESE: {
                "welcome": "Spatial-Omics GFMへようこそ",
                "model_loading": "モデルを読み込み中...",
                "analysis_complete": "解析完了",
                "error_occurred": "エラーが発生しました",
                "processing": "処理中...",
                "results": "結果",
                "settings": "設定",
                "help": "ヘルプ",
                "about": "について",
                "language": "言語",
                "performance": "パフォーマンス",
                "accuracy": "精度",
                "speed": "速度",
                "memory_usage": "メモリ使用量",
                "loading_data": "データを読み込み中...",
                "saving_results": "結果を保存中...",
                "export": "エクスポート",
                "import": "インポート",
                "cancel": "キャンセル",
                "confirm": "確認",
                "yes": "はい",
                "no": "いいえ"
            },
            SupportedLanguage.CHINESE_SIMPLIFIED: {
                "welcome": "欢迎使用Spatial-Omics GFM",
                "model_loading": "正在加载模型...",
                "analysis_complete": "分析完成",
                "error_occurred": "发生错误",
                "processing": "处理中...",
                "results": "结果",
                "settings": "设置",
                "help": "帮助",
                "about": "关于",
                "language": "语言",
                "performance": "性能",
                "accuracy": "准确度",
                "speed": "速度",
                "memory_usage": "内存使用",
                "loading_data": "正在加载数据...",
                "saving_results": "正在保存结果...",
                "export": "导出",
                "import": "导入",
                "cancel": "取消",
                "confirm": "确认",
                "yes": "是",
                "no": "否"
            },
            SupportedLanguage.KOREAN: {
                "welcome": "Spatial-Omics GFM에 오신 것을 환영합니다",
                "model_loading": "모델 로딩 중...",
                "analysis_complete": "분석 완료",
                "error_occurred": "오류가 발생했습니다",
                "processing": "처리 중...",
                "results": "결과",
                "settings": "설정",
                "help": "도움말",
                "about": "정보",
                "language": "언어",
                "performance": "성능",
                "accuracy": "정확도",
                "speed": "속도",
                "memory_usage": "메모리 사용량",
                "loading_data": "데이터 로딩 중...",
                "saving_results": "결과 저장 중...",
                "export": "내보내기",
                "import": "가져오기",
                "cancel": "취소",
                "confirm": "확인",
                "yes": "예",
                "no": "아니오"
            }
        }
        
        self.logger.info(f"✅ Initialized translations for {len(self.translations)} languages")
    
    def _initialize_localization_configs(self) -> None:
        """Initialize localization configurations"""
        self.localization_configs = {
            SupportedLanguage.ENGLISH: LocalizationConfig(
                language=SupportedLanguage.ENGLISH,
                region="US",
                date_format="%m/%d/%Y",
                number_format="1,234.56",
                currency_format="$1,234.56",
                timezone="UTC"
            ),
            SupportedLanguage.SPANISH: LocalizationConfig(
                language=SupportedLanguage.SPANISH,
                region="ES",
                date_format="%d/%m/%Y",
                number_format="1.234,56",
                currency_format="1.234,56 €",
                timezone="Europe/Madrid"
            ),
            SupportedLanguage.FRENCH: LocalizationConfig(
                language=SupportedLanguage.FRENCH,
                region="FR",
                date_format="%d/%m/%Y",
                number_format="1 234,56",
                currency_format="1 234,56 €",
                timezone="Europe/Paris"
            ),
            SupportedLanguage.GERMAN: LocalizationConfig(
                language=SupportedLanguage.GERMAN,
                region="DE",
                date_format="%d.%m.%Y",
                number_format="1.234,56",
                currency_format="1.234,56 €",
                timezone="Europe/Berlin"
            ),
            SupportedLanguage.JAPANESE: LocalizationConfig(
                language=SupportedLanguage.JAPANESE,
                region="JP",
                date_format="%Y/%m/%d",
                number_format="1,234.56",
                currency_format="¥1,234",
                timezone="Asia/Tokyo"
            ),
            SupportedLanguage.CHINESE_SIMPLIFIED: LocalizationConfig(
                language=SupportedLanguage.CHINESE_SIMPLIFIED,
                region="CN",
                date_format="%Y-%m-%d",
                number_format="1,234.56",
                currency_format="¥1,234.56",
                timezone="Asia/Shanghai"
            ),
            SupportedLanguage.KOREAN: LocalizationConfig(
                language=SupportedLanguage.KOREAN,
                region="KR",
                date_format="%Y.%m.%d",
                number_format="1,234.56",
                currency_format="₩1,234",
                timezone="Asia/Seoul"
            )
        }
    
    def set_language(self, language: SupportedLanguage) -> None:
        """Set current language"""
        if language in self.translations:
            self.current_language = language
            self.logger.info(f"🌍 Language set to {language.value}")
        else:
            self.logger.warning(f"⚠️  Language {language.value} not supported, using default")
            self.current_language = self.default_language
    
    def get_text(self, key: str, language: Optional[SupportedLanguage] = None, **kwargs) -> str:
        """
        Get translated text for key
        
        Args:
            key: Translation key
            language: Language to use (defaults to current language)
            **kwargs: Format parameters for string interpolation
            
        Returns:
            Translated text
        """
        lang = language or self.current_language
        
        # Get translation dictionary for language
        lang_dict = self.translations.get(lang, self.translations[self.default_language])
        
        # Get translated text
        text = lang_dict.get(key, key)  # Return key if translation not found
        
        # Apply string formatting if kwargs provided
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError) as e:
                self.logger.warning(f"⚠️  String formatting failed for key '{key}': {e}")
        
        return text
    
    def get_localized_config(self, language: Optional[SupportedLanguage] = None) -> LocalizationConfig:
        """Get localization configuration for language"""
        lang = language or self.current_language
        return self.localization_configs.get(lang, self.localization_configs[self.default_language])
    
    def format_number(self, number: Union[int, float], language: Optional[SupportedLanguage] = None) -> str:
        """Format number according to locale"""
        config = self.get_localized_config(language)
        
        # Simplified number formatting
        if config.language in [SupportedLanguage.GERMAN, SupportedLanguage.SPANISH]:
            # Use period for thousands, comma for decimal
            return f"{number:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        elif config.language == SupportedLanguage.FRENCH:
            # Use space for thousands, comma for decimal
            return f"{number:,.2f}".replace(",", " ").replace(".", ",")
        else:
            # Default format (US style)
            return f"{number:,.2f}"
    
    def format_currency(self, amount: Union[int, float], language: Optional[SupportedLanguage] = None) -> str:
        """Format currency according to locale"""
        config = self.get_localized_config(language)
        formatted_number = self.format_number(amount, language)
        
        # Currency symbols and positions
        currency_map = {
            SupportedLanguage.ENGLISH: f"${formatted_number}",
            SupportedLanguage.SPANISH: f"{formatted_number} €",
            SupportedLanguage.FRENCH: f"{formatted_number} €",
            SupportedLanguage.GERMAN: f"{formatted_number} €",
            SupportedLanguage.JAPANESE: f"¥{amount:,.0f}",
            SupportedLanguage.CHINESE_SIMPLIFIED: f"¥{formatted_number}",
            SupportedLanguage.KOREAN: f"₩{amount:,.0f}"
        }
        
        return currency_map.get(config.language, f"${formatted_number}")
    
    def format_date(self, date_obj, language: Optional[SupportedLanguage] = None) -> str:
        """Format date according to locale"""
        config = self.get_localized_config(language)
        return date_obj.strftime(config.date_format)
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with native names"""
        language_names = {
            SupportedLanguage.ENGLISH: "English",
            SupportedLanguage.SPANISH: "Español",
            SupportedLanguage.FRENCH: "Français",
            SupportedLanguage.GERMAN: "Deutsch",
            SupportedLanguage.JAPANESE: "日本語",
            SupportedLanguage.CHINESE_SIMPLIFIED: "中文(简体)",
            SupportedLanguage.KOREAN: "한국어",
            SupportedLanguage.ITALIAN: "Italiano",
            SupportedLanguage.PORTUGUESE: "Português",
            SupportedLanguage.DUTCH: "Nederlands"
        }
        
        return [
            {
                "code": lang.value,
                "name": language_names.get(lang, lang.value),
                "available": lang in self.translations
            }
            for lang in SupportedLanguage
        ]
    
    def detect_user_language(self, accept_language_header: str) -> SupportedLanguage:
        """
        Detect user's preferred language from Accept-Language header
        
        Args:
            accept_language_header: HTTP Accept-Language header value
            
        Returns:
            Best matching supported language
        """
        if not accept_language_header:
            return self.default_language
        
        # Parse Accept-Language header (simplified)
        languages = []
        for lang_part in accept_language_header.split(','):
            lang_part = lang_part.strip()
            if ';' in lang_part:
                lang, quality = lang_part.split(';', 1)
                try:
                    q = float(quality.split('=')[1])
                except:
                    q = 1.0
            else:
                lang = lang_part
                q = 1.0
            
            # Extract language code (e.g., 'en' from 'en-US')
            lang_code = lang.split('-')[0].lower()
            languages.append((lang_code, q))
        
        # Sort by quality score
        languages.sort(key=lambda x: x[1], reverse=True)
        
        # Find best match
        for lang_code, _ in languages:
            for supported_lang in SupportedLanguage:
                if supported_lang.value == lang_code:
                    return supported_lang
        
        return self.default_language
    
    def add_custom_translations(self, language: SupportedLanguage, translations: Dict[str, str]) -> None:
        """Add custom translations for a language"""
        if language not in self.translations:
            self.translations[language] = {}
        
        self.translations[language].update(translations)
        self.logger.info(f"✅ Added {len(translations)} custom translations for {language.value}")
    
    def export_translations(self, output_path: Path) -> None:
        """Export all translations to JSON files"""
        output_path.mkdir(parents=True, exist_ok=True)
        
        for language, translations in self.translations.items():
            file_path = output_path / f"{language.value}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📁 Exported translations to {output_path}")
    
    def import_translations(self, input_path: Path) -> None:
        """Import translations from JSON files"""
        for language in SupportedLanguage:
            file_path = input_path / f"{language.value}.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                    self.translations[language] = translations
        
        self.logger.info(f"📥 Imported translations from {input_path}")
    
    def get_translation_completeness(self) -> Dict[str, float]:
        """Get translation completeness for each language"""
        if not self.translations:
            return {}
        
        # Use English as reference
        reference_keys = set(self.translations[SupportedLanguage.ENGLISH].keys())
        
        completeness = {}
        for language, translations in self.translations.items():
            available_keys = set(translations.keys())
            completion_rate = len(available_keys & reference_keys) / len(reference_keys)
            completeness[language.value] = completion_rate
        
        return completeness
    
    def validate_translations(self) -> Dict[str, List[str]]:
        """Validate translations for consistency and completeness"""
        issues = {
            "missing_keys": [],
            "empty_values": [],
            "formatting_errors": []
        }
        
        # Use English as reference
        reference_translations = self.translations[SupportedLanguage.ENGLISH]
        
        for language, translations in self.translations.items():
            if language == SupportedLanguage.ENGLISH:
                continue
            
            # Check for missing keys
            for key in reference_translations:
                if key not in translations:
                    issues["missing_keys"].append(f"{language.value}: {key}")
                elif not translations[key].strip():
                    issues["empty_values"].append(f"{language.value}: {key}")
        
        return issues
    
    def get_rtl_languages(self) -> List[SupportedLanguage]:
        """Get list of right-to-left languages"""
        # For this implementation, no RTL languages are included
        # In a full implementation, would include Arabic, Hebrew, etc.
        return []
    
    def is_rtl_language(self, language: Optional[SupportedLanguage] = None) -> bool:
        """Check if language uses right-to-left text direction"""
        lang = language or self.current_language
        return lang in self.get_rtl_languages()


# Example usage and testing
def main():
    """Example usage of internationalization system"""
    # Initialize i18n system
    i18n = InternationalizationSystem(Path("./i18n"))
    
    # Test different languages
    languages_to_test = [
        SupportedLanguage.ENGLISH,
        SupportedLanguage.SPANISH,
        SupportedLanguage.FRENCH,
        SupportedLanguage.JAPANESE,
        SupportedLanguage.CHINESE_SIMPLIFIED
    ]
    
    for lang in languages_to_test:
        i18n.set_language(lang)
        print(f"\n=== {lang.value.upper()} ===")
        print(f"Welcome: {i18n.get_text('welcome')}")
        print(f"Processing: {i18n.get_text('processing')}")
        print(f"Results: {i18n.get_text('results')}")
        print(f"Number: {i18n.format_number(1234.56)}")
        print(f"Currency: {i18n.format_currency(1234.56)}")
    
    # Test language detection
    print(f"\nDetected language for 'en-US,en;q=0.9,es;q=0.8': {i18n.detect_user_language('en-US,en;q=0.9,es;q=0.8')}")
    
    # Get supported languages
    print(f"\nSupported languages: {i18n.get_supported_languages()}")
    
    # Get translation completeness
    print(f"\nTranslation completeness: {i18n.get_translation_completeness()}")


if __name__ == "__main__":
    main()