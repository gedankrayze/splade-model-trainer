"""
Medical domain template for the Domain Distiller.
"""

TEMPLATE = {
    "name": "Medical Domain",
    "description": "Medical literature, patient records, clinical guidelines, and medical terminology",
    "bootstrap_instructions": """Focus on key medical concepts including:
- Anatomy and physiology (organ systems, structures, functions)
- Diseases and conditions (symptoms, causes, treatments)
- Diagnostic procedures (tests, imaging, evaluation methods)
- Treatments and interventions (medications, surgeries, therapies)
- Healthcare delivery (systems, roles, processes)
- Medical specialties (cardiology, neurology, pediatrics, etc.)
- Pharmacology (drug classes, mechanisms, interactions)

Include standardized medical terminologies like ICD-10, SNOMED-CT, and MeSH terms where appropriate.

For document types, include:
- Medical research articles
- Clinical guidelines and protocols
- Electronic health records (EHR)
- Patient education materials
- Drug information and prescribing information
- Medical imaging reports
- Laboratory test results
- Medical consultation notes

For relationships, focus on anatomical relationships (e.g., organs within systems),
disease classifications (e.g., types of cardiovascular diseases), diagnostic criteria,
and treatment protocols (e.g., first-line vs. second-line treatments).
""",
    "query_complexity": {
        "simple": 0.3,  # 30% simple queries
        "intermediate": 0.5,  # 50% intermediate queries
        "complex": 0.2  # 20% complex queries
    },
    "document_length": {
        "short": 0.3,  # 30% short documents (e.g., lab results, medication info)
        "medium": 0.5,  # 50% medium documents (e.g., consultation notes, patient instructions)
        "long": 0.2  # 20% long documents (e.g., clinical guidelines, research articles)
    },
    "query_templates": [
        "What are the symptoms of {disease}?",
        "How is {disease} diagnosed?",
        "What is the standard treatment for {disease}?",
        "What are the side effects of {medication}?",
        "What is the function of the {anatomical_structure}?",
        "What is the difference between {disease_1} and {disease_2}?",
        "How does {medication} work to treat {disease}?",
        "What are the risk factors for developing {disease}?",
        "What are the complications of {disease} or {procedure}?",
        "How is {test} performed and what does it measure?"
    ],
    "document_templates": {
        "clinical_note": """PATIENT ENCOUNTER
Date: {date}
Provider: {provider}
Patient: {patient_identifier}

CHIEF COMPLAINT:
{chief_complaint}

HISTORY OF PRESENT ILLNESS:
{history}

PHYSICAL EXAMINATION:
{examination}

ASSESSMENT:
{assessment}

PLAN:
{plan}
""",
        "drug_information": """MEDICATION INFORMATION
Name: {drug_name}
Generic Name: {generic_name}
Drug Class: {drug_class}

INDICATIONS:
{indications}

MECHANISM OF ACTION:
{mechanism}

DOSAGE AND ADMINISTRATION:
{dosage}

CONTRAINDICATIONS:
{contraindications}

ADVERSE EFFECTS:
{adverse_effects}

INTERACTIONS:
{interactions}
"""
    }
}
