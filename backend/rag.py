import faiss
import numpy as np
import re
import torch
from backend.embeddings import EmbeddingModel
from sentence_transformers import util
from transformers import pipeline

# Load the lightweight model-Flan-T5-Base
try:
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  
        device=-1  # Force CPU usage
    )
    print("Flan-T5-Base Loaded Successfully")
except Exception as e:
    print(f"Error loading LLM: {e}")
    llm_pipeline = None

class RAGModel:
    def __init__(self):
        self.documents = {}
        self.embedding_model = EmbeddingModel()
        self.index = None
        self.sentences = []

    def add_document(self, doc_name, text):
        """Processes a document and stores its text embeddings."""
        self.documents[doc_name] = text
        self.sentences = self.split_text(text)
        self._update_index()

    def split_text(self, text):
        """Splits text into small, meaningful sections for better retrieval."""
        return re.split(r'(?<=\.)\s+', text)  # Splits sentences properly

    def _update_index(self):
        """Creates a FAISS index from text embeddings."""
        if not self.sentences:
            return
        
        embeddings = np.array([self.embedding_model.get_embedding(sentence) for sentence in self.sentences])
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def rerank_results(self, question, retrieved_sentences):
        """Uses an LLM re-ranker to pick the most relevant sentence."""
        question_embedding = self.embedding_model.get_embedding(question)
        similarities = [util.pytorch_cos_sim(question_embedding, self.embedding_model.get_embedding(sent))[0] for sent in retrieved_sentences]
        best_match_idx = np.argmax(similarities)  # Pick the most relevant sentence
        return retrieved_sentences[best_match_idx]

    def clean_answer(self, text):
        """Removes unnecessary formatting from the final answer."""
        text = re.sub(r'\s+', ' ', text)  # Normalize spaces
        text = re.sub(r'(\d{4}-\d{2}-\d{2})', '', text)  # Remove any date patterns like '2024-25'
        text = re.sub(r'(^\d+/\w+)', '', text, flags=re.MULTILINE)  # Remove things like "2/KALEIDOSCOPE"
        return text.strip()

    def generate_llm_answer(self, question, context):
        """Generates a well-formed answer using Flan-T5-Base."""
        if llm_pipeline is None:
            return "Error: No local LLM is available."

        prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        try:
            response = llm_pipeline(prompt, max_length=250, truncation=True, do_sample=True, temperature=0.7)
            return response[0]["generated_text"].strip()
        except Exception as e:
            return f"Error generating LLM answer: {str(e)}"

    def answer_question(self, question):
        """Retrieves the best answer for the given question and provides better context."""
        if not self.index or not self.sentences:
            return "No documents uploaded yet."

        # Get the vector embedding for the question
        question_vec = self.embedding_model.get_embedding(question).reshape(1, -1)
        
        # Retrieve the top 5 most relevant sentences
        _, I = self.index.search(question_vec, k=5)

        # Collect retrieved sentences and their neighboring sentences
        retrieved_context = []
        for idx in I[0]:
            if 0 <= idx < len(self.sentences):  # Ensure index is within range
                retrieved_context.append(self.sentences[idx])  # Add main sentence
                if idx > 0:
                    retrieved_context.append(self.sentences[idx - 1])  # Add previous sentence
                if idx < len(self.sentences) - 1:
                    retrieved_context.append(self.sentences[idx + 1])  # Add next sentence

        # Join the selected sentences as context
        full_context = " ".join(list(set(retrieved_context)))  # Remove duplicates

        # Pass  context to the LLM
        refined_answer = self.generate_llm_answer(question, full_context)

        return self.clean_answer(refined_answer)
