"""
Chat History Management using Pinecone

Stores conversation history in Pinecone for retrieval and context.
"""

import os
import hashlib
import time
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ChatHistoryManager:
    """Manages chat history using Pinecone vector database"""

    def __init__(self, mock_mode: bool = False):
        """
        Initialize ChatHistoryManager

        Args:
            mock_mode: If True, use in-memory storage instead of Pinecone (for testing)
        """
        api_key = os.getenv("PINECONE_API_KEY")
        base_index_name = os.getenv("PINECONE_INDEX_NAME")

        # Initialize OpenAI embeddings if available
        self.use_openai = False
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    openai_api_key=openai_key
                )
                self.dimension = 1536  # text-embedding-3-small dimension
                self.use_openai = True
                # Use different index name for OpenAI embeddings to avoid dimension mismatch
                index_name = base_index_name
                logger.info("Using OpenAI embeddings for chat history")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI embeddings: {e}. Using simple embeddings.")
                self.dimension = 1024
                index_name = base_index_name
        else:
            self.dimension = 1024
            index_name = base_index_name

        # Check if we should use mock mode
        if mock_mode or not api_key:
            if not api_key:
                logger.warning("PINECONE_API_KEY not set - using mock mode (in-memory storage)")
            self.mock_mode = True
            self.mock_storage = []  # In-memory storage for testing
            return

        self.mock_mode = False

        try:
            from pinecone import Pinecone, ServerlessSpec

            self.pc = Pinecone(api_key=api_key)
            self.index_name = index_name

            # Create index if it doesn't exist
            self._ensure_index_exists()

            # Get the index
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            logger.warning(f"Failed to initialize Pinecone: {e}. Using mock mode.")
            self.mock_mode = True
            self.mock_storage = []

    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        from pinecone import ServerlessSpec

        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
                )
            )
            # Wait for index to be ready
            time.sleep(1)

    def _create_embedding(self, text: str) -> List[float]:
        """
        Create embedding from text using OpenAI or fallback to simple method.
        """
        # Use OpenAI embeddings if available
        if self.use_openai:
            try:
                embedding = self.embeddings.embed_query(text)
                return embedding
            except Exception as e:
                logger.warning(f"OpenAI embedding failed: {e}. Falling back to simple embeddings.")
                # Fall through to simple embeddings

        # Fallback: Simple word-based embedding
        text_lower = text.lower()
        words = text_lower.split()

        # Create a simple bag-of-words embedding
        embedding = [0.0] * self.dimension

        for word in words:
            # Hash each word to multiple dimensions
            word_hash = hashlib.sha256(word.encode()).digest()
            for i in range(min(32, len(word_hash))):  # Use first 32 bytes
                idx = (word_hash[i] * 32 + i) % self.dimension
                embedding[idx] += 1.0

        # Normalize the embedding
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def add_message(self, user_message: str, agent_response: str):
        """Store a conversation exchange in Pinecone or mock storage"""
        message_id = hashlib.md5(
            f"{user_message}{agent_response}{time.time()}".encode()
        ).hexdigest()

        # Create embedding from user message
        embedding = self._create_embedding(user_message)

        if self.mock_mode:
            # Store in mock storage (in-memory)
            self.mock_storage.append({
                "id": message_id,
                "values": embedding,
                "metadata": {
                    "user_message": user_message[:500],
                    "agent_response": agent_response[:500],
                    "timestamp": time.time()
                }
            })
        else:
            # Store in Pinecone
            self.index.upsert(
                vectors=[{
                    "id": message_id,
                    "values": embedding,
                    "metadata": {
                        "user_message": user_message[:500],  # Limit size
                        "agent_response": agent_response[:500],
                        "timestamp": time.time()
                    }
                }]
            )

    def get_relevant_history(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, str]]:
        """Retrieve relevant conversation history based on query"""
        # Create embedding for query
        query_embedding = self._create_embedding(query)

        if self.mock_mode:
            # Mock retrieval using simple similarity
            history = []
            similarities = []

            for item in self.mock_storage:
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, item["values"])
                similarities.append((similarity, item))

            # Sort by similarity and take top_k
            similarities.sort(key=lambda x: x[0], reverse=True)

            for score, item in similarities[:top_k]:
                if score > 0.5:  # Similarity threshold
                    history.append({
                        "role": "user",
                        "content": item["metadata"].get("user_message", "")
                    })
                    history.append({
                        "role": "assistant",
                        "content": item["metadata"].get("agent_response", "")
                    })

            return history

        # Query Pinecone
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Extract conversation history
            history = []
            for match in results.matches:
                if match.score > 0.5:  # Similarity threshold
                    history.append({
                        "role": "user",
                        "content": match.metadata.get("user_message", "")
                    })
                    history.append({
                        "role": "assistant",
                        "content": match.metadata.get("agent_response", "")
                    })

            return history
        except Exception as e:
            logger.warning(f"Could not retrieve history: {e}")
            return []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def clear_history(self):
        """Clear all chat history"""
        if self.mock_mode:
            self.mock_storage = []
        else:
            self.index.delete(delete_all=True)
