"""Tiny retrieval layer: one policy file per queue (kb/<queue>.md), paragraph-level TF-IDF."""
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class KnowledgeBase:
    def __init__(self, folder="kb"):
        self.chunks = {}   # queue -> list[str]
        self.vec = {}
        self.mat = {}
        for fn in os.listdir(folder):
            if not fn.endswith(".md"):
                continue
            q = fn[:-3]
            text = open(os.path.join(folder, fn)).read()
            chunks = [c.strip().lstrip("- ").strip() for c in text.split("\n") if c.strip().startswith("- ")]
            self.chunks[q] = chunks
            self.vec[q] = TfidfVectorizer(stop_words="english").fit(chunks)
            self.mat[q] = self.vec[q].transform(chunks)

    def search(self, queue, query, k=3):
        if queue not in self.chunks:
            return []
        sims = cosine_similarity(self.vec[queue].transform([query]), self.mat[queue])[0]
        order = sims.argsort()[::-1][:k]
        return [self.chunks[queue][i] for i in order]
