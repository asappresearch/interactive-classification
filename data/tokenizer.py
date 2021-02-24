from typing import List


import nltk


class Tokenizer_nltk():    
    """
    Preprocessor that splits text on whitespace into tokens.
    Example:
        >>> preprocessor = SplitPreprocessor()
        >>> preprocessor.process('Hi how may I help you?')
        ['Hi', 'how', 'may', 'I', 'help', you?']
    """

    def process(self, text: str) -> List[str]:
        """Split text on whitespace into tokens."""
        return nltk.word_tokenize(text)