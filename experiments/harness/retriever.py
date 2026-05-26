import numpy as np
from typing import List, Dict, Any, Optional, Protocol, Union
from dataclasses import dataclass, field

@dataclass
class Document:
    """Represents a document to be indexed and retrieved."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchResult:
    """Represents a scored result from a retrieval operation."""
    document: Document
    score: float

class LexicalScorer(Protocol):
    """Protocol for a lexical scoring algorithm."""
    def fit(self, corpus: List[str]) -> None:
        ...
        
    def get_scores(self, query: str) -> List[float]:
        ...

class VectorEmbedder(Protocol):
    """Protocol for a vector embedding model."""
    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], np.ndarray]:
        ...

class BM25Scorer:
    """Default lexical scorer using rank_bm25."""
    def __init__(self, tokenizer=None):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Please install rank_bm25 to use the default lexical scorer: pip install rank_bm25")
            
        self._BM25Okapi = BM25Okapi
        self.tokenizer = tokenizer or (lambda text: text.lower().split())
        self.bm25 = None

    def fit(self, corpus: List[str]) -> None:
        tokenized_corpus = [self.tokenizer(doc) for doc in corpus]
        self.bm25 = self._BM25Okapi(tokenized_corpus)

    def get_scores(self, query: str) -> List[float]:
        if not self.bm25:
            raise ValueError("BM25Scorer has not been fitted with a corpus.")
        tokenized_query = self.tokenizer(query)
        return self.bm25.get_scores(tokenized_query)

class SentenceTransformerEmbedder:
    """Default vector embedder using sentence-transformers."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence-transformers to use the default embedder: pip install sentence-transformers")
            
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]]) -> Union[List[float], np.ndarray]:
        return self.model.encode(texts)

class LexicalRetriever:
    """Retrieves documents based on lexical similarity (e.g., BM25)."""
    def __init__(self, scorer: Optional[LexicalScorer] = None):
        # Dependency injection: use provided scorer, or default to BM25Scorer
        self.scorer = scorer or BM25Scorer()
        self.documents: List[Document] = []

    def add_documents(self, documents: List[Document]) -> None:
        self.documents.extend(documents)
        # Re-fit the lexical model with the updated corpus
        corpus = [doc.content for doc in self.documents]
        self.scorer.fit(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not self.documents:
            return []
            
        scores = self.scorer.get_scores(query)
        
        # Combine documents with scores and sort
        doc_scores = list(zip(self.documents, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [SearchResult(document=doc, score=float(score)) for doc, score in doc_scores[:top_k]]

class VectorRetriever:
    """Retrieves documents based on dense vector embeddings similarity."""
    def __init__(self, embedder: Optional[VectorEmbedder] = None):
        # Dependency injection: use provided embedder, or default to SentenceTransformerEmbedder
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
            
        new_texts = [doc.content for doc in documents]
        new_embeddings = np.array(self.embedder.encode(new_texts))
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
            self.documents.extend(documents)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.documents.extend(documents)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        if not self.documents or self.embeddings is None:
            return []
            
        query_embedding = np.array(self.embedder.encode(query))
        
        # Calculate cosine similarity
        norm_doc_embeddings = np.linalg.norm(self.embeddings, axis=1)
        norm_query_embedding = np.linalg.norm(query_embedding)
        
        # Handle zero vectors to avoid division by zero
        norm_doc_embeddings = np.where(norm_doc_embeddings == 0, 1e-10, norm_doc_embeddings)
        norm_query_embedding = 1e-10 if norm_query_embedding == 0 else norm_query_embedding
        
        similarities = np.dot(self.embeddings, query_embedding) / (norm_doc_embeddings * norm_query_embedding)
        
        # Sort and get top K
        doc_scores = list(zip(self.documents, similarities))
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [SearchResult(document=doc, score=float(score)) for doc, score in doc_scores[:top_k]]

class HybridRetriever:
    """Retrieves documents combining Lexical and Vector approaches using Reciprocal Rank Fusion."""
    def __init__(self, lexical_retriever: LexicalRetriever, vector_retriever: VectorRetriever):
        self.lexical_retriever = lexical_retriever
        self.vector_retriever = vector_retriever

    def add_documents(self, documents: List[Document]) -> None:
        """Adds documents to both underlying retrievers."""
        self.lexical_retriever.add_documents(documents)
        self.vector_retriever.add_documents(documents)

    def retrieve(self, query: str, top_k: int = 5, k_rrf: int = 60) -> List[SearchResult]:
        """Retrieve using both strategies and merge with Reciprocal Rank Fusion (RRF)."""
        # Perform retrieval on both strategies (fetch more than top_k for better fusion)
        lexical_results = self.lexical_retriever.retrieve(query, top_k=max(top_k * 2, 20))
        vector_results = self.vector_retriever.retrieve(query, top_k=max(top_k * 2, 20))
        
        # Apply Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        docs_by_id: Dict[str, Document] = {}
        
        # Rank fusion for lexical results
        for rank, result in enumerate(lexical_results):
            doc_id = result.document.id
            docs_by_id[doc_id] = result.document
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k_rrf + rank + 1)
            
        # Rank fusion for vector results
        for rank, result in enumerate(vector_results):
            doc_id = result.document.id
            docs_by_id[doc_id] = result.document
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k_rrf + rank + 1)
            
        # Sort by combined RRF score
        fused_results = [
            SearchResult(document=docs_by_id[doc_id], score=score)
            for doc_id, score in rrf_scores.items()
        ]
        
        fused_results.sort(key=lambda x: x.score, reverse=True)
        return fused_results[:top_k]
