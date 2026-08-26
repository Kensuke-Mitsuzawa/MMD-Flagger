import typing as ty

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from langchain_core.outputs import Generation


def nltk_preprocess_text(documents: ty.List[Generation]) -> list:
    """
    Performs tokenization, lowercasing, POS tagging, and lemmatization on a list of strings.
    """
    lemmatizer = WordNetLemmatizer()
    processed_documents = []

    # Helper function to convert NLTK's POS tag to WordNet's format
    def get_wordnet_pos(tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN # Default to Noun if tag is unclear

    for _gen_obj in documents:
        # 1. Lowercasing and Tokenization
        text: str = _gen_obj.text
        tokens = word_tokenize(text.lower())
        
        # 2. POS Tagging (required for accurate lemmatization)
        # tags are in the format [('token', 'POS_TAG'), ...]
        tagged_tokens = nltk.pos_tag(tokens)
        
        lemmatized_tokens = []
        for word, tag in tagged_tokens:
            # 3. Lemmatization
            # Convert NLTK tag to WordNet format
            pos = get_wordnet_pos(tag)
            
            # Perform lemmatization using the determined POS tag
            lemma = lemmatizer.lemmatize(word, pos=pos)
            lemmatized_tokens.append(lemma)
            
        processed_documents.append(lemmatized_tokens)
        
    return processed_documents