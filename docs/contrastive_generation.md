# Contrastive Pair Generation

Contrastive Pair Generation is an advanced technique in Domain Distiller for creating higher-quality negative examples that are more challenging and informative for training SPLADE models.

## Overview

Traditional approaches generate positive and negative documents independently. Contrastive Pair Generation instead creates negative documents that are deliberately similar to positive documents but with crucial differences that make them non-relevant to the query.

These "hard negatives" are much more valuable for training than randomly selected negative examples because they help the model learn subtle distinctions between relevant and non-relevant content.

## How It Works

Domain Distiller implements contrastive generation through a two-step process:

1. First, it generates a positive document that answers the query
2. Then, it creates negative documents based on that positive document, deliberately altering key aspects while maintaining similar vocabulary and structure

The result is a set of document pairs where the differences are meaningful and focused on the query intent, not just superficial differences in vocabulary or style.

## Contrastive Strategies

Domain Distiller employs multiple contrastive strategies to create diverse negative examples:

1. **Topical Shift**: Addresses the same general topic but shifts focus to an aspect that doesn't answer the query
2. **Entity Substitution**: Replaces key entities while maintaining similar structure
3. **Temporal Variance**: Changes time frames or sequences that make the document non-responsive
4. **Scope Mismatch**: Provides information that's too general or too specific to answer the query
5. **Premise Alteration**: Changes a fundamental assumption related to the query
6. **Perspective Shift**: Presents information from a different perspective that doesn't address the query

## Usage

Enable contrastive pair generation by using the `--contrastive` flag:

```bash
# For document generation
python -m src.domain_distiller.cli generate-documents \
  --queries-file ./data/legal_en_queries.json \
  --contrastive

# For the full pipeline
python -m src.domain_distiller.cli pipeline \
  --domain legal \
  --language en \
  --contrastive
```

## Example

Here's an example of contrastive pair generation for a legal query:

**Query:** "What are the elements required to prove negligence in a personal injury case?"

**Positive Document:**
```
To prove negligence in a personal injury case, the plaintiff must establish four elements: duty, breach, causation, and damages. First, the plaintiff must demonstrate that the defendant owed them a duty of care. Second, the plaintiff must show that the defendant breached this duty by failing to exercise reasonable care. Third, the plaintiff must prove causation - that the defendant's breach directly caused their injury. Finally, the plaintiff must establish that they suffered actual damages as a result. All four elements must be proven by a preponderance of the evidence for a negligence claim to succeed.
```

**Contrastive Negative Document (Topical Shift):**
```
Personal injury law encompasses various legal remedies for individuals who have suffered harm due to someone else's actions. These cases are primarily civil matters, distinct from criminal proceedings, and typically seek monetary compensation rather than punitive measures. In the United States, personal injury cases are governed by state law, with general principles derived from common law. Most personal injury claims are filed under the legal theory of negligence, though other theories like strict liability or intentional tort may apply depending on the circumstances. The statute of limitations for filing personal injury claims varies by state, typically ranging from one to six years.
```

This negative document uses similar terminology and discusses relevant concepts but doesn't actually answer what the elements of negligence are, which is what the query specifically asks for.

## Benefits for SPLADE Training

Contrastive pair generation provides several advantages:

1. **More Challenging Training Data**: Creates negative examples that are more difficult to distinguish from positive ones
2. **Better Representation Learning**: Helps models learn more nuanced content representations
3. **Reduced Shortcuts**: Prevents models from relying on superficial patterns
4. **Improved Generalization**: Better prepares models to handle real-world query ambiguity

## When to Use Contrastive Generation

Contrastive generation is particularly valuable when:

- You have limited training data (quality over quantity)
- Your domain has nuanced terminology where superficial matching can be misleading
- You want to train models that are robust against keyword-stuffing or irrelevant content
- You're finding that your model performs well on easy cases but struggles with ambiguous queries

For best results, consider using contrastive generation for at least a portion of your training data, even if generating the full dataset with this approach would be too time-consuming or expensive.
