#Use numerical arrays for comparing embedding vectors
import numpy as np
#Load a pretrained model that converts text into embeddings
from sentence_transformers import SentenceTransformer
#Reuse the Chunk structure we already created
from .chunking import Chunk

class Retriever:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        #Load the embedding model once when the retriever is created
        self.model = SentenceTransformer(model_name)

        #Store the chunks and their embeddings after indexing
        self.chunks = []
        self.embeddings = None

    def index(self, chunks: list[Chunk]):
        #Keep the original chunks so search results can return their text and source
        self.chunks = chunks

        texts = [piece.text for piece in chunks]

        #Convert every chunk into a normalized embedding vector
        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings = True,
            convert_to_numpy = True
        )
    
    def search(self, query: str, k: int = 3):
        #Make sure chunks have been indexed before trying to search
        if self.embeddings is None:
            raise RuntimeError("Index chunks before you search.")
        
        #Convert question into the same vector space as the chunks
        query_vector = self.model.encode(
            query,
            normalize_embeddings = True,
            convert_to_numpy = True
        )

        #Compare the query against every chunk embedding
        scores = self.embeddings @ query_vector

        #Rank the chunks by similarity and keep the strongest matches
        best = np.argsort(scores)[::-1][:k]

        return [(self.chunks[i], float(scores[i])) for i in best]