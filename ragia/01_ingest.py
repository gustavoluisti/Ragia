# %%

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from transformers import AutoTokenizer

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/BM25"
COLBERT_MODEL = "colbert-ir/colbertv2.0"

doc_convert = DocumentConverter()

chunker = HybridChunker(
    tokenizer=AutoTokenizer.from_pretrained(DENSE_MODEL),
    max_tokens=350,
)

# %%

doc = doc_convert.convert('../data/transcricao_pipeline_etl.md')
chunks = list(chunker.chunk(doc.document))
chunks
# %%
