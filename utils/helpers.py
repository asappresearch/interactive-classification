import os
from typing import *

import torch
from torch import FloatTensor, LongTensor, ByteTensor
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.optimizer import Optimizer
from collections import defaultdict
import numpy as np




def get_params(model: nn.Module) -> List[Parameter]:
    """
    Gets all trainable parameters in a model.

    :param model: An instance of an nn.Module.
    :return: A list all trainable parameters in the model.
    """
    return [param for param in model.parameters() if param.requires_grad]


def parameter_count(model: nn.Module) -> int:
    """
    Computes the total number of trainable parameters in a model.

    :param model: An instance of an nn.Module.
    :return: The total number of trainable parameters in the model.
    """
    return sum(param.numel() for param in get_params(model))


def compute_grad_norm(model: nn.Module) -> float:
    """
    Computes the total gradient norm from all trainable parameters in a model.

    :param model: An instance of an nn.Module.
    :return: The L2 norm of the gradient normspar from all trainable parameters in the model.
    """
    params = get_params(model)
    params = [p for p in params if isinstance(p.grad, torch.Tensor) ]

    return (sum(param.grad.norm() ** 2 for param in params) ** 0.5).item()


def compute_param_norm(model: nn.Module) -> float:
    """
    Computes the total parameter norm from all trainable parameters in a model.

    :param model: An instance of an nn.Module.
    :return: The L2 norm of the parameter norms from all trainable parameters in the model.
    """
    return (sum(param.data.norm() ** 2 for param in get_params(model)) ** 0.5).item()


def noam_step(iter_count: int,
              warmup_steps: int,
              dimensionality: int,
              scaling_factor: float):
    """
    Computes the learning rate at a given step of the noam learning rate scheduler.

    :param iter_count: The number of iterations (optimizer steps) taken so far.
    :param warmup_steps: The number of steps during which to linearly increase the learning rate before decreasing it.
    :param dimensionality: The dimensionality of the model output.
    :param scaling_factor: A scaling factor by which to multiply the learning rate computed by noam.
    """
    learning_rate = min(iter_count / warmup_steps ** 1.5, 1.0 / iter_count ** 0.5)
    learning_rate *= 1.0 / dimensionality ** 0.5
    learning_rate *= scaling_factor

    return learning_rate


def repeat_negatives(response_encodings: FloatTensor,
                     batch_size: int,
                     num_negatives: int) -> FloatTensor:
    """
    Takes a tensor containing positive and negative encodings and repeats the negative encodings so that there is one
    copy of each negative encoding for each positive encoding.

    Example:
        # [1, 1] and [2, 2] are positive encodings; [3, 3], [4, 4], and [5, 5] are negative encodings
        >>> response_encodings = FloatTensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        >>> batch_size = 2
        >>> num_negatives = 3
        >>> repeat_negatives(response_encodings, batch_size, num_negatives)
        FloatTensor([1, 1], [2, 2], [3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]])

    :param response_encodings: A FloatTensor of shape `(batch_size + num_negatives, hidden_size)` where the first
    `batch_size` elements are positive encodings and the remaining `num_negatives` elements are negative encodings.
    :param batch_size: The batch size, which is also the number of positive examples.
    :param num_negatives: The number of negatives examples.
    :return: A FloatTensor of shape `(batch_size + num_negatives * batch_size, hidden_size)` where the first
    `batch_size` elements are positive encodings, the second `batch_size` encodings are all copies of the first
    negative encoding, the third `batch_size` encodings are all copies of the second negative encoding, etc. up through
    `num_negative` negative encodings.
    """
    # Extract positive and negative response encodings
    response_encodings_positive = response_encodings[:batch_size]
    response_encodings_negative = response_encodings[batch_size:]

    # Reshape from (num_negatives, hidden_dim) to (num_negatives * batch_size, hidden_dim)
    # Ex. batch_size = 2, num_negatives = 3, hidden_size = 2
    # [[3, 3], [4, 4], [5, 5]] --> [[3, 3], [3, 3], [4, 4], [4, 4], [5, 5], [5, 5]]
    response_encodings_negative = response_encodings_negative.repeat(1, batch_size).view(num_negatives * batch_size, -1)

    # Combine positive and negative response encodings
    response_encodings = torch.cat((response_encodings_positive, response_encodings_negative))

    return response_encodings


def save_checkpoint(model: nn.Module,
                    optimizer: Optimizer,
                    epoch: int,
                    iter_count: int,
                    auc05: float,
                    config: Dict,
                    prev_save_path: str = None) -> str:
    """
    Saves a model checkpoint, including all model parameters and configurations and the current state of training.

    :param model: An nn.Module to save.
    :param optimizer: The PyTorch Optimizer being used during training.
    :param epoch: The current epoch count.
    :param iter_count: The current number of iterations (i.e. optimizer steps).
    :param auc05: The validation auc@0.05 scored by this model.
    :param config: A dictionary containing model and training configurations.
    :param prev_save_path: The path to the previous model checkpoint in order to delete it.
    :return: The path the the model checkpoint just saved.
    """
    # Generate state
    state = {
        'config': config,
        'epoch': epoch,
        'iter_count': iter_count,
        'auc05': auc05,
        # 'model': model,
        'model_state': model.state_dict(),
        'optimizer': optimizer
    }

    # Generate save path
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    save_path = os.path.join(config['checkpoint_dir'], '{}_auc05_{:.3f}_e{}.pt'.format(config['flavor'], auc05, epoch))

    # Save checkpoint and remove old checkpoint
    print('Saving checkpoint to {}'.format(save_path))
    torch.save(state, save_path)
    if prev_save_path is not None:
        os.remove(prev_save_path)

    return save_path


def load_checkpoint(checkpoint_path: str) : # Tuple[Model, Adam, int, int, float]:
    """
    Loads a model checkpoint with the model on CPU and in eval mode.

    :param checkpoint_path: Path to model checkpoint to load.
    :return: Returns a tuple with the model, optimizer, epoch number, iteration number, and auc@0.05 of the loaded model.
    """
    # Load state
    print(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=lambda storage, location: storage)
    # state['model_state'].eval()

    return state['model_state'], state['optimizer'], state['epoch'], state['iter_count'], state['auc05'], state['config']


def load_modelstate(checkpoint_path: str) :
    """
    Loads a model with the model on CPU and in eval mode.

    :param checkpoint_path: Path to model checkpoint to load.
    :return: Returns the loaded model on CPU in eval mode.
    """
    return load_checkpoint(checkpoint_path)[0]



def create_mask(lengths: LongTensor, cuda: bool = False) -> ByteTensor:
    """
    Creates a mask from a tensor of sequence lengths to mask out padding with 1s for content and 0s for padding.

    Example:
        >>> lengths = LongTensor([3, 4, 2])
        >>> create_mask(lengths)
        tensor([[1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 0]], dtype=torch.uint8)

    :param lengths: A LongTensor of shape `(batch_size)` with the length of each sequence in the batch.
    :param cuda: A boolean indicating whether to move the mask to GPU.
    :return: A ByteTensor of shape `(sequence_length, batch_size)` with 1s for content and 0s for padding.
    """
    # Get sizes
    seq_len, batch_size = lengths.max(), lengths.size(0)

    # Create length and index masks
    length_mask = lengths.unsqueeze(0).repeat(seq_len, 1)  # (seq_len, batch_size)

    index_mask = torch.arange(seq_len, dtype=torch.long).unsqueeze(1).repeat(1, batch_size)  # (seq_len, batch_size)

    # Create mask
    mask = (index_mask < length_mask)

    # Move to GPU
    if cuda:
        mask = mask.cuda()

    return mask

