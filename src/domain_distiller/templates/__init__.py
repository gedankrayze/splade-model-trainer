"""
Domain templates for the Domain Distiller.
These templates provide domain-specific configurations for bootstrapping and generating training data.
"""

import importlib
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default domain templates
DEFAULT_TEMPLATES = {
    "legal": {
        "name": "Legal Domain",
        "description": "Legal documents, contracts, case law, and legal terminology",
        "bootstrap_instructions": """Focus on key legal concepts including:
- Contract law (formation, breach, remedies)
- Tort law (negligence, strict liability, damages)
- Criminal law (elements of crimes, defenses)
- Constitutional law (rights, powers, interpretations)
- Civil procedure (jurisdiction, pleadings, discovery)

For document types, include court opinions, contracts, statutes, regulations, and legal memoranda.""",
        "query_complexity": {
            "simple": 0.3,  # 30% simple queries
            "intermediate": 0.5,  # 50% intermediate queries
            "complex": 0.2  # 20% complex queries
        },
        "document_length": {
            "short": 0.2,  # 20% short documents
            "medium": 0.6,  # 60% medium documents
            "long": 0.2  # 20% long documents
        }
    },
    "medical": {
        "name": "Medical Domain",
        "description": "Medical literature, patient records, clinical guidelines, and medical terminology",
        "bootstrap_instructions": """Focus on key medical concepts including:
- Anatomy and physiology (organ systems, structures, functions)
- Diseases and conditions (symptoms, causes, treatments)
- Diagnostic procedures (tests, imaging, evaluation methods)
- Treatments and interventions (medications, surgeries, therapies)
- Healthcare delivery (systems, roles, processes)

For document types, include medical records, clinical guidelines, research articles, patient education materials, and drug information.""",
        "query_complexity": {
            "simple": 0.3,
            "intermediate": 0.5,
            "complex": 0.2
        },
        "document_length": {
            "short": 0.3,
            "medium": 0.5,
            "long": 0.2
        }
    },
    "technical": {
        "name": "Technical Documentation",
        "description": "Software documentation, technical manuals, API references, and technical guides",
        "bootstrap_instructions": """Focus on key technical concepts including:
- Programming concepts (languages, paradigms, design patterns)
- Software architecture (components, patterns, practices)
- System administration (configuration, deployment, maintenance)
- Network infrastructure (protocols, security, topology)
- Database systems (models, queries, optimization)

For document types, include API documentation, user manuals, troubleshooting guides, installation instructions, and system specifications.""",
        "query_complexity": {
            "simple": 0.4,
            "intermediate": 0.5,
            "complex": 0.1
        },
        "document_length": {
            "short": 0.4,
            "medium": 0.5,
            "long": 0.1
        }
    },
    "finance": {
        "name": "Finance Domain",
        "description": "Financial documents, reports, market analysis, and financial terminology",
        "bootstrap_instructions": """Focus on key financial concepts including:
- Accounting principles (balance sheets, income statements, cash flow)
- Investment analysis (metrics, strategies, portfolio management)
- Corporate finance (capital structure, valuation, risk management)
- Financial markets (equities, fixed income, derivatives)
- Banking and monetary policy (interest rates, regulation, money supply)

For document types, include financial statements, annual reports, prospectuses, market analyses, and investment recommendations.""",
        "query_complexity": {
            "simple": 0.3,
            "intermediate": 0.5,
            "complex": 0.2
        },
        "document_length": {
            "short": 0.2,
            "medium": 0.6,
            "long": 0.2
        }
    }
}


def get_domain_template(domain: str, template_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a domain template by name.
    
    Args:
        domain: Domain name (legal, medical, etc.)
        template_name: Optional specific template to load
        
    Returns:
        Dictionary with domain template configuration
    """
    # First try to load from module
    try:
        if template_name:
            module_name = f".{template_name.lower()}"
        else:
            module_name = f".{domain.lower()}"
        
        module = importlib.import_module(module_name, package="src.domain_distiller.templates")
        if hasattr(module, "TEMPLATE"):
            return module.TEMPLATE
    except (ImportError, AttributeError):
        # Module not found or TEMPLATE not defined
        pass
    
    # Fall back to default templates
    if domain.lower() in DEFAULT_TEMPLATES:
        return DEFAULT_TEMPLATES[domain.lower()]
    
    # Use generic template as last resort
    logger.warning(f"No template found for domain '{domain}'. Using generic template.")
    return {
        "name": f"{domain.capitalize()} Domain",
        "description": f"Domain-specific content for {domain}",
        "bootstrap_instructions": f"Focus on key concepts relevant to the {domain} domain.",
        "query_complexity": {
            "simple": 0.3,
            "intermediate": 0.5,
            "complex": 0.2
        },
        "document_length": {
            "short": 0.3,
            "medium": 0.5,
            "long": 0.2
        }
    }
