import nltk


def setup_string_kernel() -> None:
    """Public API. Set up resources for string kernel."""
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt_tab')


if __name__ == '__main__':
    setup_string_kernel()
