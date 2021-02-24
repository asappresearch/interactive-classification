import fastText
from fastText.FastText import _FastText
from numpy import ndarray

from .embed import Embedder


class FastTextEmbedder(Embedder):
    """Class for mapping words to FastText embedding vectors."""

    def __init__(self, path: str = None, embedding: _FastText = None):
        """
        Initializes the FastText model, loading the model from disk if required.

        Note:
            1) Exactly one of `path` and `embedding` must be provided.
            2) It is preferred to use `path` to load FastText and then share the FastTextEmbedder object
            rather than sharing the raw FastText embedding model among multiple FastTextEmbedder objects.

        :param path: The path to a FastText .bin file.
        :param embedding: A loaded FastText model.
        """
        if path is None and embedding is None:
            raise ValueError('One of `path` and `embedding` must be defined.')
        elif path is not None and embedding is not None:
            raise ValueError('Only one of `path` and `embedding` may be defined.')
        elif path is not None:
            self.embedding = fastText.load_model(path)
        else:
            self.embedding = embedding

    def save(self, path: str):
        """
        Save FastText embedder model to a file.

        :param path: Save FastText .bin file to this path.
        :param clear: Clear FastText model so that this object can be pickled.
        """
        self.embedding.save_model(path)

    def get_dimension(self) -> int:
        """
        Gets the dimension of word vectors returned by the embedder.

        :return: The dimension of word vectors.
        """
        return self.embedding.get_dimension()

    def get_word_vector(self, word: str) -> ndarray:
        """
        Returns embedding vector for a word.

        :param word: A word.
        :return: A 1D numpy array of length equal to get_dimension().
        """
        return self.embedding.get_word_vector(word)
