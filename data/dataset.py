import math
from typing import *
from pprint import pformat, pprint


import numpy as np
from torch import FloatTensor, LongTensor

from data.tokenizer import Tokenizer_nltk


class Dataset:
    """
    The FaqDataset is a class which holds a data set of queries and targets for use in Autosuggest.

    Upon initialization, the query and targets are extracted from an iterator and the appropriate preprocessing,
    tokenization, and truncation are applied. Furthermore, a word to index mapping is determined and is used to
    replace all words with their corresponding word indices. Additionally, negative targets are sampled for
    each batch.

    This class is an iterator and returns batches of queries, positive/negative targets, and labels when
    iterated through. After each run through the entire data set, `recreate_dataset_if_necessary` should be called
    to sample new negative targets for the next iteration.
    """

    def __init__(self,
                 queries: Iterator[str],
                 targets: Iterator[str],
                 preprocessor: Tokenizer_nltk,
                #  batch_size: int = 64,
                 num_negatives: int =100,
                 reuse_negatives: bool = True,
                 negativepool:  Iterator[str] = np.array([]),
                 init_word_to_index: Dict[str, int] = None,
                 embedder_type: str = 'index'):
        """
        Initializes the FaqDataset, including extract queries and targets from `data_iter`, preprocessing
        them, and sampling negative targets for each batch.

        :param queries: An iterator over query.
        :param targets: An iterator over targets.
        :param preprocessor: A Preprocessor object which performs both string preprocessing and tokenization.
        :param batch_size: The batch size. This controls the number of query/positive target pairs per batch.
        :param num_negatives: The number of negative targets to sample for each positive target. These negative
        targets may be shared across a batch depending on `reuse_negatives`.
        :param reuse_negatives: True to share negative targets across a batch, False to sample separate negatives
        for each query/positive target pair in a batch.
        :param init_word_to_index: An initial word to index mapping to use when mapping words to indices. New
        word/index pairs may be added but the existing word/index pairs should not change.
        """
        self.preprocessor = preprocessor
        self.embeddertype = embedder_type
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.reuse_negatives = reuse_negatives
        self.word_to_index = init_word_to_index.copy() if init_word_to_index is not None else {}

        # Extract and preprocess queries and targets
        self.queries_text = list(queries)
        self.targets_text = list(targets)

        assert len(queries) == len(targets)
        self.queries = self._preprocess(self.queries_text)
        self.targets = self._preprocess(self.targets_text)

        if len(negativepool) == 0:
            self.negative_text = list(set(self.targets_text))
            self.negativepool =  self._preprocess(self.negative_text)
        else:
            self.negative_text = list(negativepool)
            self.negativepool = self._preprocess(self.negative_text)
        print('There are {} faq to do negative sampling'.format(len(self.negativepool)))

        # Initialize variables and create dataset by shuffling positives and sampling negatives
        self.num_examples = len(self.queries)
        self.num_batches = math.ceil(self.num_examples / self.batch_size)
        self.position = 0
        self.seed = 0
        self.negatives = []

        #self._recreate_dataset()

    def _preprocess(self, texts: List[str]) -> List[List[int]]:
        """
        Preprocesses a list of strings by applying the preprocessor/tokenizer, truncating, and mapping to word indices.

        Note:
            Also builds the `self.word_to_index` mapping from words to indices.

        :param texts: A list of strings where each string is a query or target.        
        :return: A list of lists where each element of the outer list is a query or target and each element
        of the inner list is a word index for a word in the query or target.
        """
        indices = []
        for text in texts:
            word_sequence = self.preprocessor.process(text)  # preprocess/tokenize and truncate
            index_sequence = []
            for word in word_sequence:
                index_sequence.append(self.word_to_index.setdefault(word, len(self.word_to_index)))
            if self.embeddertype == 'index':
                indices.append(index_sequence)
            if self.embeddertype == 'word':
                indices.append(word_sequence)

        return indices

    def get_word_to_index(self) -> Dict[str, int]:
        """Returns a copy of the word to index mapping."""
        return self.word_to_index.copy()

    def _recreate_dataset(self):
        """Shuffles the query/positive target pairs and samples new negative targets."""
        print('Shuffling')
        self.seed += 1
        np.random.seed(self.seed)
        perm = np.random.permutation(self.num_examples)
        self.queries = [self.queries[i] for i in perm]
        self.targets = [self.targets[i] for i in perm]


        print('Sampling negatives')
        if self.reuse_negatives:
            # self.negatives.shape == (num_negatives, num_batches)
            self.negatives = [np.random.choice(self.negativepool, size=self.num_batches)
                              for _ in range(self.num_negatives)]
        else:
            # self.negatives.shape == (num_negatives, num_examples)
            self.negatives = [np.random.choice(self.negativepool, size=self.num_examples)
                              for _ in range(self.num_negatives)]
        #print(len(self.negatives))

    def recreate_dataset_if_necessary(self):
        """If the iterator is at the beginning or end of the data set, then move the iterator to the beginning
        of the data set and shuffle the query/positive target pairs and sample new negative targets."""
        if self.position == 0 or self.position + self.batch_size > self.num_examples:
            self._recreate_dataset()
            self.position = 0

    def reshape_batch_scores(self, scores: FloatTensor) -> FloatTensor:
        """
        Reshapes a 1D tensor of scores to a 2D tensor of scores with all scores for a query on the same row.

        :param scores: A 1D FloatTensor of shape `(batch_size * (1 + num_negatives))` of scores arranged as
            [pos_1, pos_2, ..., pos_n, neg_11, neg_21, ..., neg_n1, neg_12, ..., neg_nm]
        where n is the batch size and m is the number of negatives.
        :return: A 2D FloatTensor of shape `(batch_size, 1 + num_negatives)` of scores arranged as
            [[pos_1, neg_11, neg_12, ..., neg_1m],
             [pos_2, neg_21, neg_22, ..., neg_2m],
              ...
             [pos_n, neg_n1, neg_n2, ..., neg_nm]]
        """
        return scores.view(1 + self.num_negatives, self.batch_size).t()
 
    '''
    def __iter__(self):
        return self


    def __next__(self) -> Tuple[List[List[int]], List[List[int]], List[int]]:
        """
        Returns the next batch in the data set, where a batch consists of queries, positive and negative targets,
        and labels.

        :return: A tuple with three elements:
        1) `queries`: A list of queries, where each query is a list of word indices. There are `batch_size` of them.
        2) `targets`: A list of targets, where each target is a list of word indices. The first `batch_size`
        targets are positive targets corresponding to the queries while the remaining targets are negative
        targets (there are `num_negatives` if `reuse_negatives` is True and there are `batch_size` if
        `reuse_negatives` is False).
        3) `labels:` A list with `batch_size` 1s and `batch_size * num_negatives` 0s representing the labels
        (i.e. 1 = positive, 0 = negative) of the query/target pairs.
        """
        position, batch_size = self.position, self.batch_size

        if position + batch_size <= self.num_examples:
            # Get queries and positive targets
            queries = self.queries[position:position + batch_size]
            targets_positive = self.targets[position:position + batch_size]

            # Get negative targets
            if self.reuse_negatives:
                negative_position = position // batch_size
                targets_negative = [negatives[negative_position] for negatives in self.negatives]
            else:
                targets_negative = [negative for negatives in self.negatives
                                      for negative in negatives[position:position + batch_size]]

            # Combine positive and negative targets
            targets = targets_positive + targets_negative

            # Create labels
            labels = [1] * batch_size + [0] * batch_size * self.num_negatives

            # Advance position
            self.position += self.batch_size

            return queries, targets, labels
        else:
            self.position = 0
            raise StopIteration()
    '''