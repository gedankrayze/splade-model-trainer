import logging

from src.embedder import EmbedderFactory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

splade_embedder = EmbedderFactory.create_embedder(
    embedder_type="splade",
    splade_model_dir="./fine_tuned_splade/wolf",
    logger=logger
)

embeddings = splade_embedder.encode_text("This is a test sentence for SPLADE embedding.")
print("SPLADE Embeddings:", embeddings)
