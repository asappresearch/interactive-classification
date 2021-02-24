from abc import ABC, abstractmethod

from numpy import ndarray


class Embedder(ABC):
    """Interface for an object that maps words to embedding vectors."""

    def save(self, path: str):
        """
        Optional method to save embedder model to a file, which is
        helpful when embedder model is not picklable, e.g., FastText.
        Embedders that don't need this can just rely on this default
        no-op implementation.

        :param path: Path to file for saving embedder model.
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Gets the dimension of word vectors returned by the embedder.

        :return: The dimension of word vectors.
        """
        pass

    @abstractmethod
    def get_word_vector(self, word: str) -> ndarray:
        """
        Returns embedding vector for a word.

        :param word: A word.
        :return: A 1D numpy array of length equal to self.get_dimension().
        """
        pass
