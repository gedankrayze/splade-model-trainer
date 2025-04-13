"""
Templates for SPLADE training data generation.

This module provides domain-specific and language-specific templates for the generation
of training data for SPLADE model fine-tuning.
"""

import logging
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


def get_template(domain_template: str, language: Optional[str] = None) -> Dict[str, Any]:
    """
    Get the appropriate template for the given domain and language.
    
    Args:
        domain_template: Domain template name
        language: Optional language code
        
    Returns:
        A template dictionary with system prompt and metadata
    """
    # Start with the default domain template
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


def detect_document_language(text: str) -> Optional[str]:
    """
    Detect the language of a document based on common words/markers.
    
    Args:
        text: Text content to analyze
        
    Returns:
        Language code (e.g., 'en', 'de') or None if detection failed
    """
    if not text or not isinstance(text, str):
        return None
    
    # Simple language detection based on common words
    # Take a sample of the text (first 100 words or so)
    words = text.lower().split()[:100]
    sample = " ".join(words)
    
    # Define language markers (common words) for various languages
    language_markers = {
        "en": ["the", "and", "is", "for", "in", "with", "on", "to", "at", "of", "a", "an", "this", "that"],
        "de": ["der", "die", "das", "und", "ist", "für", "in", "mit", "auf", "zu", "bei", "von", "ein", "eine"],
        "es": ["el", "la", "los", "las", "y", "es", "para", "en", "con", "por", "a", "de", "un", "una"],
        "fr": ["le", "la", "les", "et", "est", "pour", "dans", "avec", "sur", "à", "de", "un", "une"],
        # Add more languages as needed
    }
    
    # Count occurrences of marker words for each language
    lang_scores = {}
    for lang, markers in language_markers.items():
        score = sum(1 for word in words if word in markers)
        # Normalize by the number of words to get a percentage
        normalized_score = score / len(words) if words else 0
        lang_scores[lang] = normalized_score
    
    # Determine best language
    if not lang_scores:
        return None
    
    best_lang = max(lang_scores.items(), key=lambda x: x[1])
    threshold = 0.05  # At least 5% of words should be language markers
    
    if best_lang[1] >= threshold:
        logger.info(f"Detected language: {best_lang[0]} (score: {best_lang[1]:.2f})")
        return best_lang[0]
    
    logger.info("Could not confidently detect language")
    return None
