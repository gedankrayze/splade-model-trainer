"""
Domain bootstrapper for generating domain-specific knowledge.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

from .utils.api_utils import APIClient, AsyncWorkerPool
from .templates import get_domain_template
from .languages import get_language_config

# Configure logger
logger = logging.getLogger(__name__)


def load_domain_template(domain: str, template_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a domain template.
    
    Args:
        domain: Domain name (legal, medical, etc.)
        template_name: Optional specific template to load
        
    Returns:
        Dictionary with domain template configuration
    """
    template = get_domain_template(domain, template_name)
    logger.info(f"Loaded template for domain: {domain}")
    return template


def load_language_config(language: str) -> Dict[str, Any]:
    """
    Load language-specific configuration.
    
    Args:
        language: Language code (en, de, es, etc.)
        
    Returns:
        Dictionary with language configuration
    """
    language_config = get_language_config(language)
    logger.info(f"Loaded configuration for language: {language}")
    return language_config


async def generate_domain_concepts(
    api_client: APIClient,
    domain: str,
    language: str,
    domain_template: Dict[str, Any],
    language_config: Dict[str, Any],
    num_concepts: int = 50
) -> Dict[str, Any]:
    """
    Generate domain-specific concepts.
    
    Args:
        api_client: API client for generation
        domain: Domain name
        language: Language code
        domain_template: Domain template configuration
        language_config: Language configuration
        num_concepts: Number of concepts to generate
        
    Returns:
        Dictionary with domain concepts
    """
    # Prepare system prompt
    system_prompt = f"""You are an expert in {domain} domain knowledge. 
You will help create a comprehensive knowledge base for training data generation.
Your responses should be in {language_config['name']} language."""
    
    # Prepare user prompt
    user_prompt = f"""Generate a structured knowledge base for the {domain} domain in {language_config['name']} language.
This knowledge base will be used to generate training data for an information retrieval system.

Include the following sections:
1. Core concepts and terminology (at least {num_concepts} terms with definitions)
2. Common relationships between concepts
3. Typical questions users might ask in this domain
4. Standard document types in this domain
5. Domain-specific facts and examples

For the concepts, provide:
- Term name
- Definition
- Example usage or context
- Related terms

Format your response as a structured JSON object.
"""

    # Add domain-specific instructions from template
    if domain_template.get("bootstrap_instructions"):
        user_prompt += f"\n\nDomain-specific instructions:\n{domain_template['bootstrap_instructions']}"
    
    # Add language-specific instructions
    if language_config.get("bootstrap_instructions"):
        user_prompt += f"\n\nLanguage-specific instructions:\n{language_config['bootstrap_instructions']}"
    
    # Response schema
    schema = {
        "type": "object",
        "properties": {
            "domain": {"type": "string"},
            "language": {"type": "string"},
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "term": {"type": "string"},
                        "definition": {"type": "string"},
                        "example": {"type": "string"},
                        "related_terms": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["term", "definition"]
                }
            },
            "relationships": {
                "type": "array", 
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                        "examples": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "typical_questions": {
                "type": "array",
                "items": {"type": "string"}
            },
            "document_types": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                        "common_sections": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "facts": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["domain", "language", "concepts"]
    }
    
    # Generate domain concepts
    try:
        response = await api_client.structured_generation(
            prompt=user_prompt,
            system_prompt=system_prompt,
            schema=schema,
            temperature=0.3,
            max_tokens=4000
        )
        
        # Add metadata
        response["_meta"] = {
            "domain_template": domain_template.get("name", domain),
            "language": language,
            "original_domain": domain,
            "concepts_count": len(response.get("concepts", [])),
            "questions_count": len(response.get("typical_questions", [])),
        }
        
        logger.info(f"Generated {response['_meta']['concepts_count']} concepts and {response['_meta']['questions_count']} questions")
        return response
        
    except Exception as e:
        logger.error(f"Error generating domain concepts: {str(e)}")
        # Return minimal structure
        return {
            "domain": domain,
            "language": language,
            "concepts": [],
            "typical_questions": [],
            "error": str(e),
            "_meta": {
                "domain_template": domain_template.get("name", domain),
                "language": language,
                "original_domain": domain,
                "error": True
            }
        }


async def generate_additional_questions(
    api_client: APIClient,
    domain_data: Dict[str, Any],
    count: int = 50
) -> List[str]:
    """
    Generate additional questions for the domain.
    
    Args:
        api_client: API client for generation
        domain_data: Domain data with concepts
        count: Number of questions to generate
        
    Returns:
        List of additional questions
    """
    domain = domain_data.get("domain", "unknown")
    language = domain_data.get("language", "en")
    
    # Extract some concepts to use as inspiration
    concepts = domain_data.get("concepts", [])
    concept_terms = [c.get("term") for c in concepts[:10] if "term" in c]
    concept_str = ", ".join(concept_terms)
    
    # Prepare system prompt
    system_prompt = f"""You are an expert in the {domain} domain. 
Your task is to generate realistic and diverse questions that users might ask when searching for information in this domain.
Your responses should be in {language} language."""
    
    # Prepare user prompt
    user_prompt = f"""Generate {count} diverse and realistic search queries that users might enter when looking for information in the {domain} domain.

These queries should:
1. Cover a range of topics within the {domain} domain
2. Vary in complexity (simple, intermediate, complex)
3. Include different query types (factual, procedural, conceptual, comparative)
4. Be written in natural language as a user would type them
5. Be specific enough to have clear answers

Here are some concepts from this domain to inspire your questions:
{concept_str}

Format your response as a JSON array of strings, each containing a single query.
"""

    # Generate questions
    try:
        response = await api_client.structured_generation(
            prompt=user_prompt,
            system_prompt=system_prompt,
            schema={"type": "array", "items": {"type": "string"}},
            temperature=0.7,
            max_tokens=3000
        )
        
        if isinstance(response, list):
            logger.info(f"Generated {len(response)} additional questions")
            return response
        else:
            logger.warning("Response format incorrect for additional questions")
            return []
            
    except Exception as e:
        logger.error(f"Error generating additional questions: {str(e)}")
        return []


async def _process_batch(batch_id: int, domain: str, language: str, num_concepts: int, api_client: APIClient) -> Dict[str, Any]:
    """Process a single batch for domain bootstrapping."""
    try:
        # Load template and language config
        domain_template = load_domain_template(domain)
        language_config = load_language_config(language)
        
        # Generate domain concepts
        domain_data = await generate_domain_concepts(
            api_client,
            domain,
            language,
            domain_template,
            language_config,
            num_concepts=num_concepts
        )
        
        # Generate additional questions if successful
        if domain_data.get("concepts") and len(domain_data["concepts"]) > 0:
            additional_questions = await generate_additional_questions(
                api_client,
                domain_data,
                count=50  # Generate 50 additional questions
            )
            
            # Add additional questions to domain data
            existing_questions = domain_data.get("typical_questions", [])
            domain_data["typical_questions"] = list(set(existing_questions + additional_questions))
            
            # Update metadata
            if "_meta" in domain_data:
                domain_data["_meta"]["questions_count"] = len(domain_data["typical_questions"])
        
        return domain_data
        
    except Exception as e:
        logger.error(f"Error in batch {batch_id}: {str(e)}")
        return {
            "domain": domain,
            "language": language,
            "error": str(e),
            "_meta": {
                "batch_id": batch_id,
                "error": True
            }
        }


async def _bootstrap_domain_async(
    domain: str,
    language: str,
    num_concepts: int,
    api_base: str,
    api_key: str,
    model: str,
    num_workers: int,
    output_dir: str,
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """Asynchronous implementation of bootstrap_domain."""
    # Initialize API client
    api_client = APIClient(
        api_base=api_base,
        api_key=api_key,
        model=model
    )
    
    # Process batch
    logger.info(f"Bootstrapping domain knowledge for {domain} in {language}")
    result = await _process_batch(0, domain, language, num_concepts, api_client)
    
    # Save result
    if output_file:
        output_path = output_file
    else:
        output_path = os.path.join(output_dir, f"{domain}_{language}_domain.json")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved domain knowledge to {output_path}")
    
    # Close API client
    await api_client.close()
    
    return result


def bootstrap_domain(
    domain: str,
    language: str,
    num_concepts: int = 50,
    template_name: Optional[str] = None,
    api_base: str = "https://api.openai.com/v1",
    api_key: str = None,
    model: str = "gpt-4o",
    output_dir: str = "./distilled_data",
    output_file: Optional[str] = None,
    num_workers: int = 1
) -> Dict[str, Any]:
    """
    Bootstrap domain knowledge for generating training data.
    
    Args:
        domain: Domain name (legal, medical, etc.)
        language: Language code (en, de, es, etc.)
        num_concepts: Number of concepts to generate
        template_name: Optional specific template to load
        api_base: Base URL for OpenAI-compatible API
        api_key: API key
        model: Model name
        output_dir: Output directory for domain data
        output_file: Optional specific output file path
        num_workers: Number of async workers
        
    Returns:
        Dictionary with domain knowledge
    """
    # Create event loop and run async implementation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        result = loop.run_until_complete(
            _bootstrap_domain_async(
                domain=domain,
                language=language,
                num_concepts=num_concepts,
                api_base=api_base,
                api_key=api_key,
                model=model,
                num_workers=num_workers,
                output_dir=output_dir,
                output_file=output_file
            )
        )
        return result
    finally:
        loop.close()
