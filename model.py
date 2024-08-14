import random
from collections import defaultdict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class SmarterRAGModel:
    def __init__(self, text_file, chunk_size=100, overlap=50):
        with open(text_file, 'r', encoding='utf-8') as file:
            self.full_text = file.read().lower()
        
        self.chunks = self.create_chunks(chunk_size, overlap)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_vectors = self.model.encode(self.chunks)
        
        self.word_dict = defaultdict(list)
        self.build_language_model()

    def create_chunks(self, chunk_size, overlap):
        sentences = self.full_text.split('.')
        chunks = []
        for i in range(0, len(sentences), chunk_size - overlap):
            chunk = '. '.join(sentences[i:i + chunk_size]).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def build_language_model(self):
        words = self.full_text.split()
        for i in range(len(words) - 2):
            trigram = tuple(words[i:i + 2])
            self.word_dict[trigram].append(words[i + 2])

    def retrieve_relevant_chunk(self, query):
        query_vector = self.model.encode([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        most_relevant_idx = np.argmax(similarities)
        return self.chunks[most_relevant_idx]

    def generate_text(self, start_text, length=50):
        current_text = start_text.lower()
        result = current_text.split()

        for _ in range(length):
            relevant_chunk = self.retrieve_relevant_chunk(' '.join(result[-5:]))
            chunk_words = relevant_chunk.split()
            
            if len(result) >= 2:
                last_bigram = tuple(result[-2:])
                if last_bigram in self.word_dict:
                    next_word = random.choice(self.word_dict[last_bigram])
                elif result[-1] in chunk_words:
                    next_word_index = chunk_words.index(result[-1]) + 1
                    next_word = chunk_words[next_word_index] if next_word_index < len(chunk_words) else random.choice(chunk_words)
                else:
                    next_word = random.choice(chunk_words)
            else:
                next_word = random.choice(chunk_words)
            
            result.append(next_word)

        return ' '.join(result)

# Usage
model = SmarterRAGModel('war_and_peace_processed.txt')
generated_text = model.generate_text("Peace is", length=50)
print(generated_text)