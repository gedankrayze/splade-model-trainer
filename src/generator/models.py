"""
Data models for SPLADE training data generation.

This module defines Pydantic models for structured data used in the training data generator.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator


class NegativeExample(BaseModel):
    """A negative document example that doesn't answer the query."""
    document: str = Field(..., description="The negative document content that doesn't answer the query")
    explanation: str = Field(..., description="Why this document was selected as a negative example")
    strategy: Optional[str] = Field(None, description="Strategy used for contrastive generation")


class TrainingExample(BaseModel):
    """A single training example for SPLADE model training."""
    query: str = Field(..., description="A natural, specific query someone might search for")
    positive_document: str = Field(..., description="The document content that answers the query")
    negative_documents: List[NegativeExample] = Field(
        ...,
        description="List of negative examples that don't answer the query",
        json_schema_extra={"min_items": 1, "max_items": 5}
    )

    @model_validator(mode='after')
    def check_different_documents(self) -> 'TrainingExample':
        """Validate that positive and negative documents are different."""
        for neg_doc in self.negative_documents:
            if neg_doc.document == self.positive_document:
                raise ValueError("Negative document must be different from positive document")
        return self


class TrainingData(BaseModel):
    """Collection of training examples."""
    examples: List[TrainingExample]


class DocumentChunk(BaseModel):
    """A chunk of a document for processing."""
    file_path: str = Field(..., description="Path to the original file")
    file_name: str = Field(..., description="Name of the original file")
    content: str = Field(..., description="Content of the document chunk")
    extension: str = Field(..., description="File extension")
    chunk_id: Optional[int] = Field(None, description="ID of the chunk within the document")
    chunk_total: Optional[int] = Field(None, description="Total number of chunks in the document")
    error: Optional[str] = Field(None, description="Error message if document loading failed")
