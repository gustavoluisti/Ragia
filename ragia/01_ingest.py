# %%

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/BM25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

doc_convert = DocumentConverter()

chunker = HybridChunker(
    tokenizer=AutoTokenizer.from_pretrained(DENSE_MODEL),
    max_tokens=350,
)

dense_model = TextEmbedding(DENSE_MODEL)
sparse_model = SparseTextEmbedding(SPARSE_MODEL)
colbert_model = LateInteractionTextEmbedding(COLBERT_MODEL)

# %%

doc = doc_convert.convert('../data/transcricao_pipeline_etl.md')
chunks = list(chunker.chunk(doc.document))
chunks
# %%
