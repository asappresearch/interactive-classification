import os
from pprint import pformat, pprint
from typing import Dict, Iterator, Tuple, Union

import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from tqdm import trange, tqdm

from utils.aucmeter import AUCMeter


from utils.helpers import compute_grad_norm, compute_param_norm, \
    load_checkpoint,  get_params, noam_step, parameter_count, repeat_negatives, save_checkpoint


import sys


def run_epoch(dataset,
               batch_embedder,
               model,
               config: Dict,
               train: bool,
               test: bool = False,
               iter_count: int = 0,
               writer: SummaryWriter = None,
               scheduler = None,
               bird: bool=False,
               optimizer: Optimizer = None) -> Union[int, Tuple[float, float]]:
    """
    Runs a single epoch of either training or validation.
    :param dataset: An FaqDataset containing query/target pairs. May be either a training or
    a validation data set.
    :param batch_embedder: An IndexBatchEmbedder which takes the batches of word indices returned by the dataset
    and converts them to PyTorch tensors of word embeddings and sequence lengths.
    :param model: The FAQRetrieval model being trained or evaluated.
    :param config: A dictionary containing model and training configurations.
    :param train: True for a training epoch, False for a validation epoch.
    :param iter_count: The number of iterations (i.e. optimizer steps) performed so far.
    :param writer: A tensorboardX SummaryWriter for recording training and validation metrics and scalars.
    :param optimizer: A PyTorch Optimizer used to optimize the model (only necessary if `train` is True).
    :return: If `train` is True, returns the current number of iterations (i.e. optimizer steps). If `train` is False,
    returns the auc@0.05 across the (validation) data set.
    """

    # Set model mode
    if train:
        model.train()
    else:
        model.eval()

    # Recreate negatives if necessary

    dataset.recreate_dataset_if_necessary()

    # Initialize statistics
    if not train:
        auc_meter = AUCMeter()
        total_loss = total_num_correct = 0

    # Run epoch
    batch_count = 0
    max_batches_per_epoch = config['max_batches_per_epoch'] if train else float('inf')
    num_batches = min(dataset.num_batches, max_batches_per_epoch)
    for batch in tqdm(dataset, total=num_batches):
        # Noam learning rate step and model zero grad
        if train:
            iter_count += 1
            optimizer.param_groups[0]['lr'] = noam_step(iter_count,
                                                        config['warmup_steps'],
                                                        model.output_size,
                                                        config['noam_scaling_factor'])
            model.zero_grad()

        # Get batch
        queries, targets, labels = batch

        # Apply feature transformation
        query_embeddings, query_lengths = batch_embedder.embed(queries)
        target_embeddings, target_lengths = batch_embedder.embed(targets)

        # Encode queries and targets
        if bird:
            query_encodings = model.encode_context(query_embeddings, query_lengths)
            target_encodings = model.encode_class(target_embeddings, target_lengths)
        else:
            query_encodings = model.encode(query_embeddings, query_lengths)
            target_encodings = model.encode(target_embeddings, target_lengths)
        # Repeat query encodings so there is one for the positive target and one for each negative target
        query_encodings = query_encodings.repeat(1 + dataset.num_negatives, 1)

        # Repeat negative encodings if reusing negatives
        if dataset.reuse_negatives:
            target_encodings = repeat_negatives(target_encodings, dataset.batch_size, dataset.num_negatives)

        # Score query/target similarities using dot product
        scores = model.score(query_encodings, target_encodings, scoring_method='dot product')

        if config['loss_type'] =='cross_entropy':
            # Reshape scores to line up positive and negative scores for a given query on sa row
            scores_reshaped = dataset.reshape_batch_scores(scores)
            # Create target
            target = torch.zeros(dataset.batch_size).long()
            if config['cuda']:
                target = target.cuda()
            # Compute loss
            loss = F.cross_entropy(scores_reshaped, target)
            num_correct = torch.eq(scores_reshaped.argmax(dim=-1), target).sum().item()

        elif config['loss_type'] =='bce':
            target = torch.tensor(labels).float()
            if config['cuda']:
                target = target.cuda()
            loss = F.binary_cross_entropy_with_logits(scores, target)
            m = nn.Sigmoid()
            predict = torch.ge(m(scores), 0.5).long()
            num_correct = torch.eq( predict, target.long()).sum().item()/(1 + dataset.num_negatives)

        # Update statistics
        

        if not train:
            total_num_correct += num_correct
            auc_meter.add(scores.data.cpu(), np.array(labels))
            total_loss += loss.item()

        # Backprop
        if train:
            loss.backward()
            optimizer.step()

        # Print/write train results
        if train and iter_count % config['print_frequency'] == 0:
            # Compute statistics
            batch_auc_meter = AUCMeter()
            batch_auc_meter.add(scores.data.cpu(), np.array(labels))
            auc01, auc05, auc1, auc = [batch_auc_meter.value(max_fpr) for max_fpr in [0.01, 0.05, 0.1, 1.0]]

            batch_loss = loss.item()
            accuracy = num_correct / dataset.batch_size

            # Gather scalars
            scalars = [('loss', batch_loss), ('auc0.01', auc01), ('auc0.05', auc05), ('auc0.1', auc1), ('auc', auc),
                       ('accuracy', accuracy), ('grad_norm', compute_grad_norm(model)),
                       ('param_norm', compute_param_norm(model)), ('learning_rate', optimizer.param_groups[0]['lr'])]

            # Print scalars
            print('Step: {:,} '.format(iter_count) +
                  ' '.join(['{} = {:.3f}'.format(name, value) for name, value in scalars[:-1]]) + ' ' +
                  ' '.join(['{} = {:.3e}'.format(name, value) for name, value in scalars[-1:]]))

            # Write scalars to Tensorboard
            if writer is not None:
                for name, value in scalars:
                    writer.add_scalar('train/' + name, value, iter_count * dataset.batch_size)

        # Break early if desired
        batch_count += 1
        if train and batch_count == max_batches_per_epoch:
            break

    if train:
        return iter_count

    # Compute overall statistics
    auc01, auc05, auc1, auc = [auc_meter.value(max_fpr) for max_fpr in [0.01, 0.05, 0.1, 1.0]]
    loss = total_loss / batch_count
    accuracy = total_num_correct / (batch_count * dataset.batch_size)

    # Gather scalars
    scalars = [('loss', loss), ('auc0.01', auc01), ('auc0.05', auc05), ('auc0.1', auc1), ('auc', auc),
               ('accuracy', accuracy)]

    # Print scalars
    print('Validation results: ' + ' '.join(['{} = {:.3f}'.format(name, value) for name, value in scalars]))

    # Write validation scalars to Tensorboard
    if writer is not None:
        if test:  
            for name, value in scalars:
                writer.add_scalar('test/' + name, value, iter_count * dataset.batch_size)
        else :
            for name, value in scalars:
                writer.add_scalar('val/' + name, value, iter_count * dataset.batch_size)

    return accuracy