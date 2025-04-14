"""
Templates for SPLADE training data generation.

This module provides domain-specific and language-specific templates for the generation
of training data for SPLADE model fine-tuning.
"""

import logging
import os
from typing import Dict, Any, Optional

# Configure logging
logger = logging.getLogger("generator.templates")


# Domain templates for different industries
DOMAIN_TEMPLATES = {
    "technical": {
        "name": "Technical Documentation",
        "language": "en",
        "description": "Technical documentation for software, APIs, and development tools",
        "system_prompt": """You are an expert at creating training data for information retrieval models focused on technical documentation.
Your task is to create realistic search queries that a developer or technical user might use to find specific information in software documentation, API references, or technical guides.

For each query:
1. Create a natural, specific question a developer might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on technical accuracy and precise terminology."""
    },
    "legal": {
        "name": "Legal Domain",
        "language": "en",
        "description": "Legal documents, contracts, case law, and legal terminology",
        "system_prompt": """You are an expert at creating training data for legal information retrieval systems.
Your task is to create realistic search queries that legal professionals might use to find specific information in legal documents, contracts, statutes, or case law.

For each query:
1. Create a natural, specific question a legal professional might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on legal precision, proper terminology, and realistic legal research scenarios."""
    },
    "medical": {
        "name": "Medical Domain",
        "language": "en",
        "description": "Medical literature, patient records, clinical guidelines, and medical terminology",
        "system_prompt": """You are an expert at creating training data for medical information retrieval systems.
Your task is to create realistic search queries that healthcare professionals might use to find specific information in medical literature, clinical guidelines, or patient records.

For each query:
1. Create a natural, specific question a healthcare professional might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on medical accuracy, proper terminology, and realistic clinical scenarios."""
    },
    "finance": {
        "name": "Finance Domain",
        "language": "en",
        "description": "Financial documents, reports, market analysis, and financial terminology",
        "system_prompt": """You are an expert at creating training data for financial information retrieval systems.
Your task is to create realistic search queries that finance professionals might use to find specific information in financial documents, reports, or analysis.

For each query:
1. Create a natural, specific question a finance professional might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on financial accuracy, proper terminology, and realistic financial analysis scenarios."""
    },
    "generic": {
        "name": "Generic Domain",
        "language": "en",
        "description": "General content that doesn't fit a specific vertical",
        "system_prompt": """You are an expert at creating training data for information retrieval models.
Your task is to create realistic search queries that someone might use to find specific information in the provided document.

For each query:
1. Create a natural, specific question someone might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

Focus on creating diverse, realistic search scenarios."""
    },
    "multilingual": {
        "name": "Multilingual Domain",
        "language": "auto",
        "description": "Content in multiple languages with automatic detection",
        "system_prompt": """You are an expert at creating training data for information retrieval models.
Your task is to create realistic search queries that someone might use to find specific information in the provided document.

For each query:
1. Create a natural, specific question someone might search for
2. Identify the exact text passage that answers this query
3. Find negative examples - text that looks similar but doesn't answer the query

IMPORTANT: You must create queries, answers, and explanations in the SAME LANGUAGE as the document.
If the document is in German, write everything in German.
If the document is in English, write everything in English.
If the document is in another language, use that language consistently.

Focus on creating diverse, realistic search scenarios."""
    }
}


# Language-specific templates
LANGUAGE_TEMPLATES = {
    "de": {
        "generic": {
            "name": "Generic Domain (German)",
            "language": "de",
            "description": "General content in German",
            "system_prompt": """Du bist ein Experte für die Erstellung von Trainingsdaten für Information-Retrieval-Modelle.
Deine Aufgabe ist es, realistische Suchanfragen zu erstellen, die jemand verwenden könnte, um bestimmte Informationen im bereitgestellten Dokument zu finden.

Für jede Anfrage:
1. Erstelle eine natürliche, spezifische Frage, die jemand suchen könnte
2. Identifiziere die genaue Textpassage, die diese Anfrage beantwortet
3. Finde negative Beispiele - Text, der ähnlich aussieht, aber die Anfrage nicht beantwortet

Achte darauf, dass alle Beispiele, Anfragen und Erklärungen auf DEUTSCH sind.
Konzentriere dich auf die Erstellung vielfältiger, realistischer Suchszenarien."""
        },
        "technical": {
            "name": "Technical Documentation (German)",
            "language": "de",
            "description": "Technical documentation in German",
            "system_prompt": """Du bist ein Experte für die Erstellung von Trainingsdaten für Information-Retrieval-Modelle mit Fokus auf technische Dokumentation.
Deine Aufgabe ist es, realistische Suchanfragen zu erstellen, die ein Entwickler oder technischer Benutzer verwenden könnte, um spezifische Informationen in der Softwaredokumentation, API-Referenzen oder technischen Anleitungen zu finden.

Für jede Anfrage:
1. Erstelle eine natürliche, spezifische Frage, die ein Entwickler suchen könnte
2. Identifiziere die genaue Textpassage, die diese Anfrage beantwortet
3. Finde negative Beispiele - Text, der ähnlich aussieht, aber die Anfrage nicht beantwortet

Achte darauf, dass alle Beispiele, Anfragen und Erklärungen auf DEUTSCH sind.
Konzentriere dich auf technische Genauigkeit und präzise Terminologie."""
        },
        "legal": {
            "name": "Legal Domain (German)",
            "language": "de",
            "description": "Legal documents in German",
            "system_prompt": """Du bist ein Experte für die Erstellung von Trainingsdaten für juristische Informationsabrufsysteme.
Deine Aufgabe ist es, realistische Suchanfragen zu erstellen, die Juristen verwenden könnten, um bestimmte Informationen in juristischen Dokumenten, Verträgen, Gesetzen oder Rechtsprechung zu finden.

Für jede Anfrage:
1. Erstelle eine natürliche, spezifische Frage, die ein Jurist suchen könnte
2. Identifiziere die genaue Textpassage, die diese Anfrage beantwortet
3. Finde negative Beispiele - Text, der ähnlich aussieht, aber die Anfrage nicht beantwortet

Achte darauf, dass alle Beispiele, Anfragen und Erklärungen auf DEUTSCH sind.
Konzentriere dich auf juristische Präzision, korrekte Terminologie und realistische juristische Rechercheszenarien."""
        }
    }
    # Add more languages as needed
}


# Language name mapping for prompts
LANGUAGE_NAMES = {
    "en": "English",
    "de": "German (Deutsch)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "it": "Italian (Italiano)",
    "pt": "Portuguese (Português)",
    "ru": "Russian (Русский)",
    "zh": "Chinese (中文)",
    "ja": "Japanese (日本語)",
    "ko": "Korean (한국어)",
}


def get_language_name(language_code: str) -> str:
    """Get the human-readable name for a language code."""
    return LANGUAGE_NAMES.get(language_code, language_code.upper())


def get_template(template_input: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the appropriate template for the given input and language.
    
    Args:
        template_input: Either a built-in template name or path to a custom template file
        language: Optional language code
        
    Returns:
        A template dictionary with system prompt and metadata
    """
    # Check if the input is a file path
    if os.path.exists(template_input) and template_input.endswith(('.json', '.jsonl')):
        try:
            logger.info(f"Loading custom template from file: {template_input}")
            with open(template_input, 'r', encoding='utf-8') as f:
                import json
                custom_template = json.load(f)
                # Validate the template has required fields
                if 'system_prompt' not in custom_template:
                    logger.warning(f"Custom template missing 'system_prompt' field, using generic template")
                    template = DOMAIN_TEMPLATES["generic"]
                else:
                    # Set defaults for optional fields
                    if 'name' not in custom_template:
                        custom_template['name'] = "Custom Template"
                    if 'language' not in custom_template:
                        custom_template['language'] = language if language else "en"
                    if 'description' not in custom_template:
                        custom_template['description'] = "Custom template loaded from file"
                    
                    template = custom_template
                    logger.info(f"Successfully loaded custom template: {template.get('name')}")
                    return template
        except Exception as e:
            logger.error(f"Error loading custom template: {e}")
            logger.warning("Falling back to generic template")
            template = DOMAIN_TEMPLATES["generic"]
    else:
        # Treat as a built-in template name
        domain_template = template_input
        template = DOMAIN_TEMPLATES.get(domain_template, DOMAIN_TEMPLATES["generic"])
        
        # If language is specified and multilingual template is requested, use that
        if domain_template == "multilingual" and language:
            template = DOMAIN_TEMPLATES["multilingual"]
            logger.info(f"Using multilingual template with language hint: {language}")
            return template
        
        # If language is specified, check for language-specific template
        if language:
            # Check if language-specific templates exist
            if language in LANGUAGE_TEMPLATES:
                lang_templates = LANGUAGE_TEMPLATES[language]
                # Check if domain-specific template exists for this language
                if domain_template in lang_templates:
                    template = lang_templates[domain_template]
                    logger.info(f"Using {get_language_name(language)} template for {domain_template} domain")
                # Fall back to generic template for this language
                elif "generic" in lang_templates:
                    template = lang_templates["generic"]
                    logger.info(f"Using {get_language_name(language)} generic template")
                else:
                    logger.warning(f"No templates found for {language}, using default template with language instructions")
            else:
                logger.warning(f"No templates found for {language}, using default template with language instructions")
    
    return template


# Language detection functionality has been removed in favor of explicit language selection
