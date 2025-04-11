"""
German language configuration for the Domain Distiller.
"""

LANGUAGE = {
    "name": "German",
    "code": "de",
    "description": "German language (Deutsch)",
    "bootstrap_instructions": """
Generate all content in fluent, professional German (Deutsch).

Pay special attention to:
1. Use proper German grammar and case structure
2. Use appropriate compound nouns (Komposita) common in German technical language
3. Include the gender article with nouns (der/die/das)
4. Use formal language style ("Sie" form) for instructions and questions
5. Correctly use German-specific punctuation and capitalization rules
6. Incorporate both German terms and internationally used terms where appropriate (with German translation)

Ensure all content follows German linguistic patterns rather than direct translations from English.
""",
    "query_patterns": [
        "Was ist {concept}?",
        "Wie funktioniert {process}?",
        "Definieren Sie {term}",
        "Erklären Sie {concept}",
        "Was sind die Unterschiede zwischen {concept1} und {concept2}?",
        "Nennen Sie Beispiele für {concept}",
        "Was sind die Vorteile von {concept}?",
        "Wie kann ich {action}?",
        "Warum ist {concept} wichtig?",
        "Wann sollte man {action}?",
        "Welche Arten von {concept} gibt es?",
        "Wie lässt sich {concept} optimieren?",
        "Welche Rolle spielt {concept} bei {process}?",
        "Was muss ich beachten, wenn ich {action} möchte?",
        "Wie hängen {concept1} und {concept2} zusammen?"
    ],
    "document_patterns": {
        "definition": """
{term} ({genus})

Definition:
{definition}

Verwendung:
{usage}

Verwandte Begriffe:
{related_terms}
""",
        "instruction": """
{title}

Zweck:
{purpose}

Benötigte Materialien:
{materials}

Vorgehensweise:
{steps}

Hinweise:
{notes}
"""
    }
}
