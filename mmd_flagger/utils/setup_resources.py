import nltk
from sentence_transformers import SentenceTransformer


def setup_string_kernel() -> None:
    """Public API. Set up resources for string kernel."""
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt_tab')


def setup_sentence_embedding_model() -> None:
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')



if __name__ == '__main__':
    setup_string_kernel()
    setup_sentence_embedding_model()
