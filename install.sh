#!/bin/bash

# Gedank Rayze SPLADE Model Trainer Installation Script

echo "Installing Gedank Rayze SPLADE Model Trainer..."

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing package in development mode..."
pip install -e .

# Update imports if needed
echo "Updating imports..."
python update_imports.py

echo "Installation completed successfully!"
echo "You can now use the Gedank Rayze SPLADE Model Trainer."
echo ""
echo "To activate the virtual environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To train a model, run:"
echo "  python -m src.train_splade --train-file training_data.json --output-dir ./fine_tuned_splade"
echo ""
echo "To test a model, run:"
echo "  python -m tests.code.test_queries --model-dir ./fine_tuned_splade --docs-file documents.json"
