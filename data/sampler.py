from typing import Tuple, Iterator, Dict, List, Sequence
import numpy as np

import torch
from torch import FloatTensor
from torch.nn.utils.rnn import pad_sequence

from flambe.sampler.sampler import Sampler
from flambe.field import TextField, LabelField

from isrs_flambe.utils.dataloader import read_intentdescription, read_keywords_cat


class ISRSSampler(Sampler):
    """Implements a RetrievalSampler object."""

    def __init__(self,
                 text_field,
                 label_field,
                 datatype,
                 datadir: str=None,
                 batch_size: int = 10,
                 num_negatives: int = 15,
                 # reuse_negatives: bool = True,
                 negativepool:  Iterator[str] = np.array([]),
                 batch_first: bool = False) -> None:
        """Initialize the RetrievalSampler.
        Parameters
        ----------
        """
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.batch_first = batch_first
        self.n_batch = 1
        self.text_field = text_field
        self.label_field = label_field
        self.datatype =datatype

        alllabels, alldescription = read_intentdescription(datadir, datatype=self.datatype)
        self.mapping = {x[0]:x[1] for x in zip(alllabels, alldescription)}
        # alldescription, self.mapping = read_intentdescription(datatype='protset_newanno')
        # all_targets_text = targets = [list(self.text_field.process(desc).numpy()) for desc in alldescription]
        all_targets_text = targets = [self.text_field.process(desc) for desc in alldescription]
        all_targets_text = pad_sequence(all_targets_text, batch_first=self.batch_first, padding_value=0)
        self.all_targets = all_targets_text

    def transform_label(self, targets: List[torch.Tensor], label_to_text: bool=True):
        if not label_to_text:
            return targets
        
        label_rev_vocab = { v:k for k,v in self.label_field.vocab.items()}
        targets = [self.text_field.process(self.mapping[label_rev_vocab[tag]]) for tag in targets]
        return targets


    def sample(self,
               data: Sequence[Sequence[torch.Tensor]],
               n_epochs: int = 1) -> Tuple[List[List[int]], List[List[int]], List[int]]:
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

        all_queries = []
        all_targets = []
        for q, t in data:
            all_queries.append(q)
            all_targets.append(t)

        print(f'sampler size: {len(all_queries)}')


        self.n_batch = int(np.ceil(data.__len__() / self.batch_size))
        print("n_batch:", self.n_batch)

        for i in range(self.n_batch):
            # position = i * self.batch_size
            # queries = all_queries[position:position + self.batch_size]
            # targets = all_targets[position:position + self.batch_size]
            sample_index = np.random.choice(len(all_queries), self.batch_size)
            queries = [all_queries[i] for i in sample_index]
            targets_label = [all_targets[i] for i in sample_index]

            # targets = self.transform_label(targets_label)

            # labels = np.arange(len(queries))

            # queriess = np.array(queries)
            all_targets_text = self.all_targets
            queries = pad_sequence(queries, batch_first=self.batch_first, padding_value=0)

            # targets, queries, labels = torch.tensor(targets), torch.tensor(labels)
            # print(queries[:5])
            # print(len(all_targets_text))


            targets_label = torch.tensor(targets_label)
            yield (queries, all_targets_text, targets_label)



    def length(self, data: Sequence[Sequence[torch.Tensor]]) -> int:
        """Return the number of batches in the sampler.
        Parameters
        ----------
        data: Sequence[Sequence[torch.Tensor, ...]]
            The input data to sample from
        Returns
        -------
        int
            The number of batches that would be created per epoch
        """
        return self.n_batch
