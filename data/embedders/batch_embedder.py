from abc import ABC, abstractmethod
from collections import OrderedDict
import logging
import sys
from typing import Dict, List, Set, Tuple, Union

import torch
from torch import FloatTensor, LongTensor
import torch.nn as nn

from .embed import Embedder

logger = logging.getLogger("batch_embedder")
#logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class BatchEmbedder(ABC):
    """
    Abstract class for optimized embedding of a batch of sequences of tokens, with zero padding added.

    The BatchEmbedder contains an `embed` method which takes in a batch of tokens and returns two tensors, one
    containing the token embeddings and one containing the length of each sequence of tokens. A batch of tokens
    is represented as a list of lists where elements of the outer list are sequences in the batch and the elements
    of the inner batch are tokens in the sequence. If tokens are strings (i.e. words), then the WordBatchEmbedder
    should be used. If tokens are ints (i.e. pre-computed word indices), then the IndexBatchEmbedder should be used
    since it is faster. In either case, the BatchEmbedder provides zero padding.

    The BatchEmbedder optimizes the token to embedding transformation process by caching embeddings in an
    nn.Embedding layer. Because this layer maps all tokens to embeddings simultaneously rather than sequentially
    as with a loop, this transformation is extremely fast. Furthermore, if a GPU is available, the embedding may be
    performed on GPU for a further speedup. The IndexBatchEmbedder and WordBatchEmbedder use slightly different
    caching schemes, so see their docstrings for details.
    """

    def __init__(self,
                 embedder: Embedder,
                 word_to_index: OrderedDict,
                 init_cache_size: int = 1,
                 cuda: bool = False):
        """
        Initializes the BatchEmbedder and adds all words in `word_to_index` to the nn.Embedding cache.

        :param embedder: An object for mapping words to embedding vectors.
        :param word_to_index: A dictionary mapping words to the indices at which they will be stored in the cache.
        :param init_cache_size: The initial size of the cache (if larger than word_to_index).
        :param cuda: A bool indicating whether to perform embedding on GPU instead of CPU.
        """
        if set(range(len(word_to_index))) != set(word_to_index.values()):
            raise ValueError('word_to_index must map to every index in range(0, len(word_to_index))')

        self.embedder = embedder
        self.word_to_index = word_to_index
        self.init_cache_size = init_cache_size
        self.is_cuda = cuda

        # Note: padding index always lives at the largest index in `self.embedding_layer`
        self.padding_idx = len(self.word_to_index)

        # Initialize embedding layer
        num_embeddings = max(self.init_cache_size, len(self.word_to_index)) + 1  # + 1 for padding index
        self.embedding_layer = nn.Embedding(num_embeddings=num_embeddings,
                                            embedding_dim=self.embedder.get_dimension(),
                                            padding_idx=self.padding_idx)
        self.embedding_layer.weight.requires_grad = False

        # Optionally move embedding to GPU
        if self.is_cuda:
            self.embedding_layer = self.embedding_layer.cuda()

        # Fill cache with words in `self.word_to_index`
        self._warm_cache()

    def get_dimension(self) -> int:
        """
        Returns the word embedding dimension.

        :return: The dimension of word vectors.
        """
        return self.embedder.get_dimension()

    def cuda(self) -> 'BatchEmbedder':
        """
        Moves the cached embeddings to GPU and changes `self.embed` to return GPU tensors.

        :return: self
        """
        self.is_cuda = True
        self.embedding_layer = self.embedding_layer.cuda()

        return self

    def cpu(self) -> 'BatchEmbedder':
        """
        Moves the cached embeddings to CPU and changes `self.embed` to return CPU tensors.

        :return: self
        """
        self.is_cuda = False
        self.embedding_layer = self.embedding_layer.cpu()

        return self

    def _warm_cache(self):
        """
        Uses `self.embedder` to fill the cache with word vectors for all words in `self.word_to_index`.

        Note:
            Each word vector is stored in the cache at the index corresponding to the word in `self.word_to_index`.
        """
        for word, index in self.word_to_index.items():
            self.embedding_layer.weight.data[index].copy_(torch.from_numpy(self.embedder.get_word_vector(word)))

    def _embed(self, index_batch: List[List[int]]) -> Tuple[FloatTensor, LongTensor]:
        """
        Embeds a batch of indices and returns a tensors with the embeddings and sequence lengths.

        :param batch: A list of lists of ints. Each element of the outer list is a single sequence in the batch.
        Each element of the inner list is a word index (as specified by `self.word_to_index`).
        :return: A tuple of (FloatTensor, LongTensor) with shapes `(sequence_length, batch_size, embedding_size)` and
        `(batch_size)` respectively. The FloatTensor contains the embedded batch, with zero padding added as necessary.
        The LongTensor contains the length of each sequence in the batch.
        """
        # Get lengths
        lengths = LongTensor([len(sequence) for sequence in index_batch])
        # TODO: switch to pytorch version 0.4.1 and add .item()
        seq_len = lengths.max() if torch.__version__[:3] == '0.3' else lengths.max().item()

        # Add padding
        index_batch = [sequence + [self.padding_idx] * (seq_len - len(sequence)) for sequence in index_batch]

        # Cast to tensor
        index_batch = LongTensor(index_batch)  # (batch_size, sequence_length)

        # Optionally move to GPU
        if self.is_cuda:
            index_batch = index_batch.cuda()

        # Get embeddings
        batch = self.embedding_layer(index_batch)  # (batch_size, sequence_length, embedding_size)

        # Transpose to (sequence_length, batch_size, embedding_size)
        batch = batch.transpose(0, 1)

        return batch, lengths

    @abstractmethod
    def embed(self, batch: Union[List[List[int]], List[List[str]]]) -> Tuple[FloatTensor, LongTensor]:
        """
        Embeds a batch of indices or words and returns a tensors with the embeddings and sequence lengths.

        :param batch: A list of lists of ints or strings. Each element of the outer list is a single sequence in the
        batch. Each element of the inner list is either a word or a word's index (as specified by `self.word_to_index`).
        :return: A tuple of (FloatTensor, LongTensor) with shapes `(sequence_length, batch_size, embedding_size)` and
        `(batch_size)` respectively. The FloatTensor contains the embedded batch, with zero padding added as necessary.
        The LongTensor contains the length of each sequence in the batch.
        """
        pass


class IndexBatchEmbedder(BatchEmbedder):
    """
    The IndexBatchEmbedder is a BatchEmbedder optimized for operating on batches of sequences of word indices.

    In order to use the IndexBatchEmbedder, a word to index dictionary must be pre-computed and passed to the
    IndexBatchEmbedder upon initialization, and it cannot be changed afterwards. During initialization, the
    IndexBatchEmbedder will load the word embedding vector for each word in the `word_to_index` dictionary into
    an nn.Embedding layer, which then acts as a cache for quick embedding lookup. When performing the embedding,
    use the `embed` method to embed a batch of sequences of word indexes.
    """

    def __init__(self,
                 embedder: Embedder,
                 word_to_index: Dict[str, int],
                 cuda: bool = False):
        """
        Initializes the IndexBatchEmbedder and adds all words in `word_to_index` to the nn.Embedding cache.

        :param embedder: An object for mapping words to embedding vectors.
        :param word_to_index: A dictionary mapping words to the indices at which they will be stored in the cache.
        :param cuda: A bool indicating whether to perform embedding on GPU instead of CPU.
        """
        super(IndexBatchEmbedder, self).__init__(embedder, OrderedDict(word_to_index), cuda=cuda)

    def embed(self, index_batch: List[List[int]]) -> Tuple[FloatTensor, LongTensor]:
        """
        Embeds a batch of indices and returns a tensors with the embeddings and sequence lengths.

        :param index_batch: A list of lists of ints. Each element of the outer list is a single sequence in the batch.
        Each element of the inner list is a word index (as specified by `self.word_to_index`).
        :return: A tuple of (FloatTensor, LongTensor) with shapes `(sequence_length, batch_size, embedding_size)` and
        `(batch_size)` respectively. The FloatTensor contains the embedded batch, with zero padding added as necessary.
        The LongTensor contains the length of each sequence in the batch.
        """
        return self._embed(index_batch)


class WordBatchEmbedder(BatchEmbedder):
    """
    The WordBatchEmbedder is a BatchEmbedder optimized for operation on batches of sequences of words.

    Like the IndexBatchEmbedder, the WordBatchEmbedder uses an nn.Embedding layer as a cache for fast embedding lookup.
    However, unlike the IndexBatchEmbedder, the WordBatchEmbedder's nn.Embedding layer does not have to be filled upon
    initialization and it may grow dynamically while being used.

    If a set of words is provided during initialization, the embeddings for these words will be added to the cache
    upon initialization for quick access when using `embed`. If no words are provided during initialization, the cache
    starts empty.

    As the `embed` method is used to embed words, the embeddings of any words not currently in the cache are looked up
    and added to the cache before use. If the cache is filled, it doubles in size up to a provided `max_cache_size`.
    Once the cache reaches its maximum size, it uses a least recently used (LRU) caching scheme.
    """

    def __init__(self,
                 embedder: Embedder,
                 words: Set[str] = None,
                 init_cache_size: int = 1,
                 max_cache_size: int = None,
                 cuda: bool = False):
        """
        Initializes the WordBatchEmbedder and adds all words in `words` to the nn.Embedding cache.

        :param embedder: An object for mapping words to embedding vectors.
        :param words: A set of words to add to the cache.
        :param init_cache_size: The initial size of the cache (if larger than word_to_index).
        :param max_cache_size: The maximum size of the cache.
        :param cuda: A bool indicating whether to perform embedding on GPU instead of CPU.
        """
        self.max_cache_size = max_cache_size or sys.maxsize
        if init_cache_size > self.max_cache_size:
            raise ValueError('Initial cache size cannot be larger than maximum cache size')
        words = words or set()
        word_to_index = self._init_word_to_index(words)
        super(WordBatchEmbedder, self).__init__(embedder, word_to_index, init_cache_size=init_cache_size, cuda=cuda)
        self.cached_words = set(self.word_to_index.keys())

        self.num_hits = 0
        self.num_misses = 0
        self.num_batches = 0

    def _init_word_to_index(self, words: Set[str]) -> OrderedDict:
        """
        Creates a mapping from each word in the set provided to a unique index in the range(0, len(words)).

        Note:
            1) The order in the returned OrderedDict is arbitrary.
            2) If more words are provided than `self.max_cache_size`, then `self.max_cache_size` words will be
            arbitrarily chosen and added to the mapping and the other words will be ignored.

        :param words: A set of words.
        :return: An OrderedDict mapping each word to a unique index in the range(0, len(words)).
        """
        word_to_index = OrderedDict()
        for word in words:
            if len(word_to_index) >= self.max_cache_size:
                break
            word_to_index[word] = len(word_to_index)

        return word_to_index

    def _expand_cache(self, size: int):
        """
        Expands the nn.Embedding cache to double the currently required size, with a maximum of `self.max_cache_size`.
        
        :param size: The desired size of the cache (i.e. number of words currently in cache + number of new words
        to add to cache).
        """

        # Don't expand if already at max cache size
        if self.embedding_layer.num_embeddings - 1 >= self.max_cache_size:  # - 1 for pad
            return

        # Expand embedding to double required size (up to maximum cache size)
        new_num_embeddings = min(size * 2, self.max_cache_size + 1)  # + 1 for pad
        logger.debug('Expanding embedding cache from {:,} to {:,}'.format(self.embedding_layer.num_embeddings,
                                                                          new_num_embeddings),
                     extra={"log_id": "batch_embedder_cache_expansion"})
        self.padding_idx = new_num_embeddings - 1
        new_embedding_layer = nn.Embedding(num_embeddings=new_num_embeddings,
                                           embedding_dim=self.embedding_layer.embedding_dim,
                                           padding_idx=self.padding_idx)
        new_embedding_layer.weight.requires_grad = False

        # Copy over old weights and delete old embedding layer
        new_embedding_layer.weight[:self.embedding_layer.num_embeddings] = self.embedding_layer.weight
        del self.embedding_layer
        if self.is_cuda:
            new_embedding_layer = new_embedding_layer.cuda()
        self.embedding_layer = new_embedding_layer

    def _cache_words(self, words: Set[str]) -> Dict[str, int]:
        """
        Ensures that all words provided are in the cache and returns info about cache.

        Words which are already in the cache are moved to the front of the cache to prevent removal under the LRU
        caching scheme. Words which are not in the cache are added to the cache, which may require removing the least
        recently used words which are not present in the provided set of words.

        :param words: The set of words to put (or keep) in the cache.
        :return: A dictionary containing information about the current state of the cache.
        """
        # Raise error if more words than can fit in cache
        if len(words) > self.max_cache_size:
            raise ValueError('Cannot fit entire batch of words into cache.'
                             'Found {} unique words but cache only fits {} words.'.format(len(words),
                                                                                          self.max_cache_size))

        # Determine new words and expand embedding layer if necessary
        new_words = words.difference(self.cached_words)
        size = len(self.cached_words) + len(new_words) + 1  # + 1 for padding index

        if size > self.embedding_layer.num_embeddings:
            self._expand_cache(size)

        # Move all words that are about to be used to the front of the cache
        # so that they are not removed when inserting new words
        current_words = words.intersection(self.cached_words)
        for word in current_words:
            self.word_to_index.move_to_end(word)

        # Add new words to cache using LRU caching scheme
        for word in new_words:
            # Cache is not full so use new index
            if len(self.word_to_index) < self.embedding_layer.num_embeddings - 1:  # - 1 to keep room for padding index
                index = len(self.word_to_index)
            # Cache is full so get LRU index
            else:
                old_word, index = self.word_to_index.popitem(last=False)
                self.cached_words.remove(old_word)

            # Add word to cache
            self.word_to_index[word] = index
            self.cached_words.add(word)
            self.embedding_layer.weight.data[index].copy_(torch.from_numpy(self.embedder.get_word_vector(word)))

        # Info about cache
        cache_info = {
            'Hits': len(current_words),
            'Percent Hits': len(current_words) / len(words),
            'Misses': len(new_words),
            'Percent Misses': len(new_words) / len(words),
            'Cache Size': self.embedding_layer.num_embeddings,
            'Number of Words in Cache': len(self.cached_words)
        }

        return cache_info

    def embed_with_info(self, word_batch: List[List[str]]) -> Tuple[Tuple[FloatTensor, LongTensor], Dict[str, int]]:
        """
        Embeds a batch of words and returns a tensors with the embeddings and sequence lengths and gets cache info.

        :param word_batch: A list of lists of strings. Each element of the outer list is a single sequence in the batch.
        Each element of the inner list is a word.
        :return: A tuple with two elements:
        1) A tuple of (FloatTensor, LongTensor) with shapes `(sequence_length, batch_size, embedding_size)` and
        `(batch_size)` respectively. The FloatTensor contains the embedded batch, with zero padding added as necessary.
        The LongTensor contains the length of each sequence in the batch.
        2) A dictionary containing information about the current state of the cache.
        """
        # Add words to cache
        words = {word for sequence in word_batch for word in sequence}
        cache_info = self._cache_words(words)

        # Convert to indices
        index_batch = [[self.word_to_index[word] for word in sequence] for sequence in word_batch]

        return self._embed(index_batch), cache_info

    def embed(self, word_batch: List[List[str]]) -> Tuple[FloatTensor, LongTensor]:
        """
        Embeds a batch of words and returns a tensors with the embeddings and sequence lengths.

        :param word_batch: A list of lists of strings. Each element of the outer list is a single sequence in the batch.
        Each element of the inner list is a word.
        :return: A tuple of (FloatTensor, LongTensor) with shapes `(sequence_length, batch_size, embedding_size)` and
        `(batch_size)` respectively. The FloatTensor contains the embedded batch, with zero padding added as necessary.
        The LongTensor contains the length of each sequence in the batch.
        """
        return self.embed_with_info(word_batch)[0]
