# Custom Templates Implementation Notes

## Overview

Implemented a flexible custom template system for training data generation that significantly enhances domain-specific adaptation capabilities. This system allows users to create specialized templates without modifying the codebase.

## Changes Made

### Core Functionality Changes

1. **Replaced `domain_template` with `template` Parameter**
   - Modified `src/generate_training_data.py` to use a more flexible `--template` parameter
   - New parameter accepts both built-in template names and file paths
   - Updated all function calls and references throughout the codebase

2. **Enhanced Template Loading System**
   - Rewrote `get_template` function in `src/generator/templates.py` to handle file paths
   - Added JSON loading with proper error handling
   - Implemented validation for required template fields
   - Added fallback to built-in templates when custom templates fail to load

3. **Removed Automatic Language Detection**
   - Removed `detect_document_language` function which was creating overhead
   - Simplified language handling by relying on explicit `--language` parameter
   - Removed unused `sample` variable that was flagged in the code review

4. **Domain Distiller Integration**
   - Updated the Domain Distiller CLI to also support custom templates
   - Ensured compatibility between generate_training_data.py and domain_distiller

### Documentation and Examples

1. **Created Custom Templates Documentation**
   - Added comprehensive `docs/custom_templates.md` with explanation and examples
   - Detailed the template structure, best practices, and usage patterns
   - Included examples for various domains and languages

2. **Established Templates Directory**
   - Created a root-level `templates/` directory for custom templates
   - Added 10 domain-specific templates, including:
     - Cold Chain Management
     - Healthcare IT Systems
     - Sustainable Manufacturing 
     - Cybersecurity Incident Response
     - Financial Risk Management
     - Renewable Energy Projects
     - HVAC Documentation (English and German)
     - HVAC Technical (English and German)
   - Added a README.md in the templates directory

3. **Updated Main Documentation**
   - Added custom templates section to README.md
   - Updated domain_distiller.md to reference custom templates
   - Referenced the new feature in appropriate places

### Task Runner Improvements

1. **Enhanced Taskfiles**
   - Updated Taskfile.yaml to support template and language parameters
   - Added comprehensive multiline comments for each task
   - Added parameter descriptions and usage examples

2. **Added New Task**
   - Created `prepare-with-openai-lang` task for generating language-specific data
   - Added proper parameter handling and documentation
   - Removed obsolete `prepare-with-ollama-auto` task

## Technical Details

### Custom Template Structure

```json
{
  "name": "Template Name",
  "language": "en",
  "description": "Brief description of the template domain",
  "system_prompt": "Detailed instructions for the LLM..."
}
```

The `system_prompt` field contains domain-specific instructions that guide the LLM in generating appropriate training data.

### Template Loading Logic

Templates are loaded with the following priority:
1. If path exists and is a JSON file, load as custom template
2. If loading fails, fall back to built-in template
3. Apply language-specific template if available
4. Use generic template as final fallback

### Error Handling

- Added validation for custom templates
- Created helpful error messages
- Implemented graceful fallbacks when templates fail to load

## Engineering Decisions

1. **Why JSON Format?**
   - Chose JSON for custom templates because it's universal and easy to edit
   - Allows for simple schema validation
   - Easily extensible for future template features

2. **Removal of Auto-Detection**
   - Auto-detection was error-prone and added unnecessary complexity
   - Explicit language selection provides more predictable results
   - Simplified the code and reduced dependencies

3. **File-based Approach vs Code Modification**
   - File-based approach allows users to create templates without code changes
   - Prevents need to modify source code for each new domain
   - Allows easier sharing of templates between users

## Future Work

1. **Template Repository**
   - Could establish a central repository for community-contributed templates
   - Add template search and discovery features

2. **Template Versioning**
   - Add version field to templates for compatibility tracking
   - Implement migration tools for updating templates

3. **Template Composition**
   - Allow templates to inherit and extend from other templates
   - Enable more complex template relationships

4. **UI for Template Creation**
   - Develop a simple UI for creating and editing templates
   - Include template testing and validation tools

5. **Metrics for Template Quality**
   - Implement analytics to measure training data quality by template
   - Create feedback loop for template improvement

## Testing Notes

Templates have been tested with:
- Different LLM providers (OpenAI, Ollama)
- Multiple languages (English primary focus)
- Various domains with specialized terminology
- Both Task runner and direct CLI usage

All custom templates successfully generate high-quality, domain-appropriate training data in initial tests.
