"""
Legal domain template for the Domain Distiller.
"""

TEMPLATE = {
    "name": "Legal Domain",
    "description": "Legal documents, contracts, case law, and legal terminology",
    "bootstrap_instructions": """Focus on key legal concepts including:
- Contract law (formation, breach, remedies)
- Tort law (negligence, strict liability, damages)
- Criminal law (elements of crimes, defenses)
- Constitutional law (rights, powers, interpretations)
- Civil procedure (jurisdiction, pleadings, discovery)
- Property law (real property, intellectual property)
- Business law (corporations, partnerships, agency)

Include specialized legal terminology and Latin phrases commonly used in legal documents.

For document types, include:
- Court opinions
- Contracts and agreements
- Statutes and regulations
- Legal memoranda
- Briefs and motions
- Legal opinions
- Legislative history

For relationships, focus on legal hierarchies (e.g., Supreme Court > Circuit Courts > District Courts),
doctrinal relationships (e.g., contract formation requires offer, acceptance, consideration), and
procedural sequences (e.g., complaint > answer > discovery > trial).
""",
    "query_complexity": {
        "simple": 0.2,  # 20% simple queries
        "intermediate": 0.5,  # 50% intermediate queries
        "complex": 0.3  # 30% complex queries
    },
    "document_length": {
        "short": 0.1,  # 10% short documents (e.g., legal definitions)
        "medium": 0.6,  # 60% medium documents (e.g., contract clauses, statute sections)
        "long": 0.3  # 30% long documents (e.g., full opinions, comprehensive analyses)
    },
    "query_templates": [
        "What are the elements of {legal_concept}?",
        "How does {jurisdiction} define {legal_term}?",
        "What is the difference between {legal_concept_1} and {legal_concept_2}?",
        "What are the requirements for {legal_action} in {jurisdiction}?",
        "Under what circumstances can {legal_action} be {outcome}?",
        "What is the standard of review for {legal_issue}?",
        "What remedies are available for {legal_wrong}?",
        "How has {legal_doctrine} evolved since {case_name}?",
        "What defenses can be raised against a claim of {legal_claim}?",
        "What is the statute of limitations for {legal_action} in {jurisdiction}?"
    ],
    "document_templates": {
        "case_summary": """CASE SUMMARY: {case_name}
CITATION: {citation}
COURT: {court}
DATE: {date}

FACTS:
{facts}

ISSUE:
{issue}

HOLDING:
{holding}

REASONING:
{reasoning}

DISPOSITION:
{disposition}
""",
        "statute_section": """TITLE: {title}
SECTION: {section}
EFFECTIVE DATE: {effective_date}

TEXT:
{text}

DEFINITIONS:
{definitions}

EXCEPTIONS:
{exceptions}

CROSS-REFERENCES:
{cross_references}
"""
    }
}
