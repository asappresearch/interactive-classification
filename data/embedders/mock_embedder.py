import numpy as np
from numpy import ndarray

from .embed import Embedder


class MockEmbedder(Embedder):
    """
    Mock implementation of a word embedder which converts a string of 3 ints to a vector.

    Example:
        >>> embedder = MockEmbedder()
        >>> embedder.get_word_vector('123')
        array([1, 2, 3])
    """

    dimension = 3

    def get_dimension(self) -> int:
        """
        Gets the dimension of word vectors returned by the embedder.

        :return: The dimension of word vectors.
        """
        return self.dimension

    def get_word_vector(self, word: str) -> ndarray:
        """
        Returns embedding vector for a word.

        :param word: A word.
        :return: A 1D numpy array of length equal to get_dimension().
        """
        if len(word) != self.dimension:
            raise ValueError('word must be a string of {} ints.'.format(self.dimension))

        return np.array([int(char) for char in word])
