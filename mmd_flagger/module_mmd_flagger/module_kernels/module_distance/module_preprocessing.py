import joblib
import os
import time
import typing as ty
import nltk
from pathlib import Path
import itertools

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from pydantic import BaseModel

import logging
logger = logging.getLogger()



def nltk_preprocess_text(documents: ty.List[str]) -> ty.List[ty.List[str]]:
    """
    Performs tokenization, lowercasing, POS tagging, and lemmatization on a list of strings.

    Return: [ [token] ]
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

    for text in documents:
        # 1. Lowercasing and Tokenization
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


class StringFeatureDictionary(BaseModel):
    vocab_dict: ty.Dict[str, int]
    unk_default: str = 'unk'

    def get_size(self) -> int:
        return len(self.vocab_dict)

    # --- Serialization (Saving Efficiently to Disk) ---
    def save_vocab(self, filename: Path):
        """Saves the dictionary using joblib compression."""
        # Joblib is great for compressing and saving large Python objects like dicts.
        joblib.dump(self.vocab_dict, filename, compress=3) # compress=3 (medium compression level)
        logger.debug(f"✅ Vocab dictionary saved efficiently to {filename}.")
        logger.debug(f"File size: {os.path.getsize(filename) / 1024:.2f} KB")

    @classmethod
    def from_documents(cls, documents: ty.List[str], is_proprocessed: bool = False, unk_default: str = 'unk') -> "StringFeatureDictionary":
        """
        documents: a sequence of tokens.
        """
        if is_proprocessed:
            pass
        else:
            # ["1st sentence", "2nd sentence"] -> ["1st", "2nd", "sentence"]
            seq_double_tokens = nltk_preprocess_text(documents)
            tokens = list(itertools.chain.from_iterable(seq_double_tokens))
            documents = tokens
        # end if
        
        vocab_dict = {unk_default: 0}
        for f_id, f in enumerate(sorted(set(documents)), start=1):
            vocab_dict[f] = f_id
        # end for

        return StringFeatureDictionary(vocab_dict=vocab_dict)

    @classmethod
    def load_vocab(cls, filename: Path, unk_default: str = 'unk') -> "StringFeatureDictionary":
        """Loads the dictionary from the compressed file."""
        assert filename.exists()
        start_time = time.time()
        loaded_dict = joblib.load(filename)
        end_time = time.time()
        
        assert unk_default in loaded_dict, f"{unk_default} is not in the given vocab. It must be."
        logger.debug(f"✅ Vocab dictionary loaded in {end_time - start_time:.4f} seconds.")
        return StringFeatureDictionary(vocab_dict=loaded_dict)


# --- Execution ---
if __name__ == "__main__":
    VOCAB_SIZE = 15000  # Example: > 10,000 size
    