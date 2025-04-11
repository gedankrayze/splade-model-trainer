"""
Language configurations for the Domain Distiller.
These configurations provide language-specific settings for bootstrapping and generating training data.
"""

import importlib
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default language configurations
DEFAULT_LANGUAGES = {
    "en": {
        "name": "English",
        "code": "en",
        "description": "English language",
        "bootstrap_instructions": "Generate all content in fluent, professional English.",
        "query_patterns": [
            "What is {concept}?",
            "How does {process} work?",
            "Define {term}",
            "Explain {concept}",
            "What are the differences between {concept1} and {concept2}?",
            "What are examples of {concept}?",
            "What are the benefits of {concept}?",
            "How can I {action}?",
            "Why is {concept} important?",
            "When should I {action}?"
        ]
    },
    "de": {
        "name": "German",
        "code": "de",
        "description": "German language (Deutsch)",
        "bootstrap_instructions": "Generate all content in fluent, professional German (Deutsch).",
        "query_patterns": [
            "Was ist {concept}?",
            "Wie funktioniert {process}?",
            "Definiere {term}",
            "Erkläre {concept}",
            "Was sind die Unterschiede zwischen {concept1} und {concept2}?",
            "Was sind Beispiele für {concept}?",
            "Was sind die Vorteile von {concept}?",
            "Wie kann ich {action}?",
            "Warum ist {concept} wichtig?",
            "Wann sollte ich {action}?"
        ]
    },
    "es": {
        "name": "Spanish",
        "code": "es",
        "description": "Spanish language (Español)",
        "bootstrap_instructions": "Generate all content in fluent, professional Spanish (Español).",
        "query_patterns": [
            "¿Qué es {concept}?",
            "¿Cómo funciona {process}?",
            "Define {term}",
            "Explica {concept}",
            "¿Cuáles son las diferencias entre {concept1} y {concept2}?",
            "¿Cuáles son ejemplos de {concept}?",
            "¿Cuáles son los beneficios de {concept}?",
            "¿Cómo puedo {action}?",
            "¿Por qué es importante {concept}?",
            "¿Cuándo debo {action}?"
        ]
    },
    "fr": {
        "name": "French",
        "code": "fr",
        "description": "French language (Français)",
        "bootstrap_instructions": "Generate all content in fluent, professional French (Français).",
        "query_patterns": [
            "Qu'est-ce que {concept}?",
            "Comment fonctionne {process}?",
            "Définis {term}",
            "Explique {concept}",
            "Quelles sont les différences entre {concept1} et {concept2}?",
            "Quels sont des exemples de {concept}?",
            "Quels sont les avantages de {concept}?",
            "Comment puis-je {action}?",
            "Pourquoi {concept} est-il important?",
            "Quand devrais-je {action}?"
        ]
    }
}


def get_language_config(language_code: str) -> Dict[str, Any]:
    """
    Get a language configuration by language code.
    
    Args:
        language_code: ISO language code (en, de, es, fr, etc.)
        
    Returns:
        Dictionary with language configuration
    """
    # Normalize language code
    language_code = language_code.lower()
    
    # First try to load from module
    try:
        module_name = f".{language_code}"
        module = importlib.import_module(module_name, package="src.domain_distiller.languages")
        if hasattr(module, "LANGUAGE"):
            return module.LANGUAGE
    except (ImportError, AttributeError):
        # Module not found or LANGUAGE not defined
        pass
    
    # Fall back to default languages
    if language_code in DEFAULT_LANGUAGES:
        return DEFAULT_LANGUAGES[language_code]
    
    # Use English as last resort
    logger.warning(f"No configuration found for language '{language_code}'. Using English.")
    return DEFAULT_LANGUAGES["en"]
