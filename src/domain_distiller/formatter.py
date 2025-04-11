"""
Dataset formatter for converting generated data into various output formats.
"""

import json
import logging
import os
import random
from typing import Dict, Any, List, Optional, Tuple, Set

# Configure logger
logger = logging.getLogger(__name__)


def load_dataset(dataset_file: str) -> List[Dict[str, Any]]:
    """
    Load dataset from file.
    
    Args:
        dataset_file: Path to dataset file
        
    Returns:
        List of dataset examples
    """
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} examples from {dataset_file}")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise ValueError(f"Failed to load dataset from {dataset_file}: {str(e)}")


def format_to_splade(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format dataset to SPLADE training format.
    
    Args:
        dataset: List of dataset examples
        
    Returns:
        List of formatted examples in SPLADE format
    """
    formatted_data = []
    
    for example in dataset:
        query = example.get("query", "")
        positive_document = example.get("positive_document", "")
        negative_documents = example.get("negative_documents", [])
        
        # Format negative documents
        negative_docs = []
        explanations = []
        
        for neg_doc in negative_documents:
            if isinstance(neg_doc, dict) and "document" in neg_doc:
                negative_docs.append(neg_doc["document"])
                explanation = neg_doc.get("explanation", "No explanation provided")
                explanations.append(explanation)
            elif isinstance(neg_doc, str):
                negative_docs.append(neg_doc)
                explanations.append("No explanation provided")
        
        # Create SPLADE example
        splade_example = {
            "query": query,
            "positive_document": positive_document,
            "negative_documents": negative_docs,
            "explanations": explanations
        }
        
        # Copy metadata if available
        if "_meta" in example:
            splade_example["_meta"] = example["_meta"]
        
        formatted_data.append(splade_example)
    
    return formatted_data


def format_to_json(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format dataset to generic JSON format.
    
    Args:
        dataset: List of dataset examples
        
    Returns:
        List of formatted examples in generic JSON format
    """
    # For JSON format, we can use the dataset as is, with some normalization
    formatted_data = []
    
    for example in dataset:
        formatted_example = {
            "query": example.get("query", ""),
            "documents": []
        }
        
        # Add positive document
        positive_document = example.get("positive_document", "")
        positive_explanation = example.get("positive_explanation", "")
        
        formatted_example["documents"].append({
            "text": positive_document,
            "is_relevant": True,
            "explanation": positive_explanation
        })
        
        # Add negative documents
        negative_documents = example.get("negative_documents", [])
        
        for neg_doc in negative_documents:
            if isinstance(neg_doc, dict) and "document" in neg_doc:
                formatted_example["documents"].append({
                    "text": neg_doc["document"],
                    "is_relevant": False,
                    "explanation": neg_doc.get("explanation", "")
                })
            elif isinstance(neg_doc, str):
                formatted_example["documents"].append({
                    "text": neg_doc,
                    "is_relevant": False,
                    "explanation": ""
                })
        
        # Copy metadata if available
        if "_meta" in example:
            formatted_example["_meta"] = example["_meta"]
        
        formatted_data.append(formatted_example)
    
    return formatted_data


def format_to_jsonl(dataset: List[Dict[str, Any]]) -> str:
    """
    Format dataset to JSONL format (one JSON object per line).
    
    Args:
        dataset: List of dataset examples
        
    Returns:
        String with JSONL-formatted data
    """
    # Convert to the generic JSON format first
    json_data = format_to_json(dataset)
    
    # Convert to JSONL
    jsonl_lines = []
    for example in json_data:
        jsonl_lines.append(json.dumps(example, ensure_ascii=False))
    
    return "\n".join(jsonl_lines)


def format_to_csv(dataset: List[Dict[str, Any]]) -> str:
    """
    Format dataset to CSV format.
    
    Args:
        dataset: List of dataset examples
        
    Returns:
        String with CSV-formatted data
    """
    import csv
    from io import StringIO
    
    # Create CSV file in memory
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(["query", "document", "is_relevant"])
    
    # Write data
    for example in dataset:
        query = example.get("query", "")
        positive_document = example.get("positive_document", "")
        
        # Write positive example
        writer.writerow([query, positive_document, 1])
        
        # Write negative examples
        negative_documents = example.get("negative_documents", [])
        
        for neg_doc in negative_documents:
            if isinstance(neg_doc, dict) and "document" in neg_doc:
                writer.writerow([query, neg_doc["document"], 0])
            elif isinstance(neg_doc, str):
                writer.writerow([query, neg_doc, 0])
    
    return output.getvalue()


def format_to_tsv(dataset: List[Dict[str, Any]]) -> str:
    """
    Format dataset to TSV format.
    
    Args:
        dataset: List of dataset examples
        
    Returns:
        String with TSV-formatted data
    """
    import csv
    from io import StringIO
    
    # Create TSV file in memory
    output = StringIO()
    writer = csv.writer(output, delimiter="\t")
    
    # Write header
    writer.writerow(["query", "document", "is_relevant"])
    
    # Write data
    for example in dataset:
        query = example.get("query", "")
        positive_document = example.get("positive_document", "")
        
        # Write positive example
        writer.writerow([query, positive_document, 1])
        
        # Write negative examples
        negative_documents = example.get("negative_documents", [])
        
        for neg_doc in negative_documents:
            if isinstance(neg_doc, dict) and "document" in neg_doc:
                writer.writerow([query, neg_doc["document"], 0])
            elif isinstance(neg_doc, str):
                writer.writerow([query, neg_doc, 0])
    
    return output.getvalue()


def split_dataset(
    dataset: List[Dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into training, validation, and test sets.
    
    Args:
        dataset: List of dataset examples
        train_ratio: Ratio of examples to use for training
        val_ratio: Ratio of examples to use for validation
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    # Shuffle dataset
    shuffled_data = dataset.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split sizes
    total = len(shuffled_data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split dataset
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test")
    return train_data, val_data, test_data


def format_dataset(
    dataset_file: str,
    output_format: str = "splade",
    split: bool = False,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None
) -> Dict[str, str]:
    """
    Format dataset to specified format.
    
    Args:
        dataset_file: Path to dataset file
        output_format: Output format (splade, json, jsonl, csv, tsv)
        split: Whether to split into train/val/test sets
        train_ratio: Ratio of examples to use for training
        val_ratio: Ratio of examples to use for validation
        output_dir: Output directory
        output_file: Optional specific output file path
        
    Returns:
        Dictionary with paths to output files
    """
    # Load dataset
    dataset = load_dataset(dataset_file)
    
    # Extract domain and language for filenames
    domain = "unknown"
    language = "en"
    if dataset and len(dataset) > 0 and "_meta" in dataset[0]:
        meta = dataset[0].get("_meta", {})
        domain = meta.get("domain", domain)
        language = meta.get("language", language)
    
    # Determine output directory
    if not output_dir:
        output_dir = os.path.dirname(dataset_file)
    
    # Set up output paths
    output_paths = {}
    
    # Determine base output path
    if output_file:
        base_output_path = output_file
    else:
        if output_format == "csv" or output_format == "tsv":
            base_output_path = os.path.join(output_dir, f"{domain}_{language}.{output_format}")
        else:
            base_output_path = os.path.join(output_dir, f"{domain}_{language}_{output_format}.json")
    
    # Format dataset
    if split:
        # Split dataset
        train_data, val_data, test_data = split_dataset(dataset, train_ratio, val_ratio)
        
        # Get base name and extension
        base_name, ext = os.path.splitext(base_output_path)
        
        # Format and save each split
        if output_format == "splade":
            # SPLADE format
            train_formatted = format_to_splade(train_data)
            val_formatted = format_to_splade(val_data)
            test_formatted = format_to_splade(test_data)
            
            # Save train data
            train_path = f"{base_name}_train{ext}"
            with open(train_path, "w", encoding="utf-8") as f:
                json.dump(train_formatted, f, ensure_ascii=False, indent=2)
            output_paths["train"] = train_path
            
            # Save validation data
            val_path = f"{base_name}_val{ext}"
            with open(val_path, "w", encoding="utf-8") as f:
                json.dump(val_formatted, f, ensure_ascii=False, indent=2)
            output_paths["val"] = val_path
            
            # Save test data
            test_path = f"{base_name}_test{ext}"
            with open(test_path, "w", encoding="utf-8") as f:
                json.dump(test_formatted, f, ensure_ascii=False, indent=2)
            output_paths["test"] = test_path
            
        elif output_format == "json":
            # JSON format
            train_formatted = format_to_json(train_data)
            val_formatted = format_to_json(val_data)
            test_formatted = format_to_json(test_data)
            
            # Save train data
            train_path = f"{base_name}_train{ext}"
            with open(train_path, "w", encoding="utf-8") as f:
                json.dump(train_formatted, f, ensure_ascii=False, indent=2)
            output_paths["train"] = train_path
            
            # Save validation data
            val_path = f"{base_name}_val{ext}"
            with open(val_path, "w", encoding="utf-8") as f:
                json.dump(val_formatted, f, ensure_ascii=False, indent=2)
            output_paths["val"] = val_path
            
            # Save test data
            test_path = f"{base_name}_test{ext}"
            with open(test_path, "w", encoding="utf-8") as f:
                json.dump(test_formatted, f, ensure_ascii=False, indent=2)
            output_paths["test"] = test_path
            
        elif output_format == "jsonl":
            # JSONL format
            train_formatted = format_to_jsonl(train_data)
            val_formatted = format_to_jsonl(val_data)
            test_formatted = format_to_jsonl(test_data)
            
            # Save train data
            train_path = f"{base_name}_train.jsonl"
            with open(train_path, "w", encoding="utf-8") as f:
                f.write(train_formatted)
            output_paths["train"] = train_path
            
            # Save validation data
            val_path = f"{base_name}_val.jsonl"
            with open(val_path, "w", encoding="utf-8") as f:
                f.write(val_formatted)
            output_paths["val"] = val_path
            
            # Save test data
            test_path = f"{base_name}_test.jsonl"
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(test_formatted)
            output_paths["test"] = test_path
            
        elif output_format == "csv":
            # CSV format
            train_formatted = format_to_csv(train_data)
            val_formatted = format_to_csv(val_data)
            test_formatted = format_to_csv(test_data)
            
            # Save train data
            train_path = f"{base_name}_train.csv"
            with open(train_path, "w", encoding="utf-8") as f:
                f.write(train_formatted)
            output_paths["train"] = train_path
            
            # Save validation data
            val_path = f"{base_name}_val.csv"
            with open(val_path, "w", encoding="utf-8") as f:
                f.write(val_formatted)
            output_paths["val"] = val_path
            
            # Save test data
            test_path = f"{base_name}_test.csv"
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(test_formatted)
            output_paths["test"] = test_path
            
        elif output_format == "tsv":
            # TSV format
            train_formatted = format_to_tsv(train_data)
            val_formatted = format_to_tsv(val_data)
            test_formatted = format_to_tsv(test_data)
            
            # Save train data
            train_path = f"{base_name}_train.tsv"
            with open(train_path, "w", encoding="utf-8") as f:
                f.write(train_formatted)
            output_paths["train"] = train_path
            
            # Save validation data
            val_path = f"{base_name}_val.tsv"
            with open(val_path, "w", encoding="utf-8") as f:
                f.write(val_formatted)
            output_paths["val"] = val_path
            
            # Save test data
            test_path = f"{base_name}_test.tsv"
            with open(test_path, "w", encoding="utf-8") as f:
                f.write(test_formatted)
            output_paths["test"] = test_path
    else:
        # Format entire dataset
        if output_format == "splade":
            # SPLADE format
            formatted_data = format_to_splade(dataset)
            
            # Save data
            with open(base_output_path, "w", encoding="utf-8") as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            output_paths["all"] = base_output_path
            
        elif output_format == "json":
            # JSON format
            formatted_data = format_to_json(dataset)
            
            # Save data
            with open(base_output_path, "w", encoding="utf-8") as f:
                json.dump(formatted_data, f, ensure_ascii=False, indent=2)
            output_paths["all"] = base_output_path
            
        elif output_format == "jsonl":
            # JSONL format
            formatted_data = format_to_jsonl(dataset)
            
            # Use .jsonl extension
            if not base_output_path.endswith(".jsonl"):
                base_output_path = os.path.splitext(base_output_path)[0] + ".jsonl"
            
            # Save data
            with open(base_output_path, "w", encoding="utf-8") as f:
                f.write(formatted_data)
            output_paths["all"] = base_output_path
            
        elif output_format == "csv":
            # CSV format
            formatted_data = format_to_csv(dataset)
            
            # Save data
            with open(base_output_path, "w", encoding="utf-8") as f:
                f.write(formatted_data)
            output_paths["all"] = base_output_path
            
        elif output_format == "tsv":
            # TSV format
            formatted_data = format_to_tsv(dataset)
            
            # Save data
            with open(base_output_path, "w", encoding="utf-8") as f:
                f.write(formatted_data)
            output_paths["all"] = base_output_path
    
    logger.info(f"Formatted dataset to {output_format} format")
    for split_name, path in output_paths.items():
        logger.info(f"Saved {split_name} data to {path}")
    
    return output_paths
