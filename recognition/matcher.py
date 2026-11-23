
import pickle
import numpy as np


class FaceMatcher:

    def __init__(self, pickle_path, threshold=0.45):

        # Load existing face database
        with open(pickle_path, "rb") as file:
            data = pickle.load(file)

        self.names = data["names"]
        self.embeddings = np.asarray(
            data["embeddings"],
            dtype=np.float32
        )

        if len(self.names) != len(self.embeddings):
            raise ValueError("Names and embeddings count mismatch")

        if self.embeddings.ndim != 2 or self.embeddings.shape[1] != 512:
            raise ValueError("Expected embeddings with shape (N, 512)")

        # Normalize stored embeddings
        norms = np.linalg.norm(
            self.embeddings, axis=1, keepdims=True
        )

        self.embeddings = self.embeddings / np.maximum(norms, 1e-12)

        self.threshold = threshold

        print(f"Loaded {len(self.names)} registered faces")

    def compare(self, live_embedding):

        live_embedding = np.asarray(
            live_embedding,
            dtype=np.float32
        ).reshape(-1)

        if live_embedding.shape[0] != 512:
            raise ValueError("Live embedding must contain 512 features")

        norm = np.linalg.norm(live_embedding)

        if norm < 1e-12:
            return None, 0.0

        live_embedding = live_embedding / norm

        # Calculate cosine similarity against all saved faces
        similarities = self.embeddings @ live_embedding

        # Find the best match
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        if best_score >= self.threshold:
            return self.names[best_index], best_score

        return None, best_score
