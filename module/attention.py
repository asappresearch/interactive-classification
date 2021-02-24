from typing import *

import torch
from torch import FloatTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentiveLayer(nn.Module):
    """Class which applies a self-attentive layer to a batch of sequences of vectors."""

    def __init__(self, hidden_size: int, config: Dict):
        """
        Initializes the SelfAttentiveLayer.

        :param hidden_size: The hidden size of the input vectors.
        :param config: A dictionary containing model configurations.
        """
        super(SelfAttentiveLayer, self).__init__()

        self.hidden_size = hidden_size
        self.num_attention_units = config['num_attention_units']
        self.num_attention_heads = config['num_attention_heads']

        self.dropout = config['dropout']
        self.dropout_layer = nn.Dropout(config['dropout'])
        self.weight_1 = nn.Linear(self.hidden_size, self.num_attention_units, bias=False)
        self.weight_2 = nn.Linear(self.num_attention_units, self.num_attention_heads, bias=False)
        self.weight_1.weight.data.uniform_(-0.1, 0.1)
        self.weight_2.weight.data.uniform_(-0.1, 0.1)

    def compute_output_and_attention_weights(self,
                                             batch: FloatTensor,
                                             mask: FloatTensor) -> FloatTensor:
        """
        Applies self-attention to each sequence in the batch and returns both the output and the attention weights.

        :param batch: A FloatTensor of shape `(sequence_length, batch_size, hidden_size)`.
        :param mask: A FloatTensor of shape `(sequence_length, batch_size)` with 1s for content and 0s for padding.
        :return: A tuple of (FloatTensor, FloatTensor) with shapes `(batch_size, hidden_size)` and
        `(batch_size, num_attention_heads, sequence_length)` respectively. The first FloatTensor is the output of
        the attention layer. The second FloatTensor contains the attention weights.
        """
        seq_len, batch_size, hidden_size = batch.size()

        batch = torch.transpose(batch, 0, 1).contiguous()  # (batch_size, seq_len, hidden_size)
        compressed_embeddings = batch.view(-1, hidden_size)  # (batch_size * seq_len, hidden_size)

        hbar = torch.tanh(self.weight_1(self.dropout_layer(compressed_embeddings)))  # (batch_size * seq_len, units)

        alphas = self.weight_2(hbar).view(batch_size, seq_len, -1)  # (batch_size, seq_len, heads)
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # (batch_size, heads, seq_len)

        if mask is not None:
            # TODO: switch to pytorch version 0.4.1 and delete this:
            if torch.__version__[:3] == '0.3':
                mask = Variable(mask, volatile=batch.volatile)

            mask = mask.transpose(0, 1)  # (batch_size, seq_len)
            mask = mask.unsqueeze(1)  # (batch_size, 1, seq_len)
            alphas = alphas * mask + (1 - mask) * (-1e+20)

        alphas = F.softmax(alphas.view(-1, seq_len), dim=-1)  # (batch_size * heads, seq_len)
        alphas = alphas.view(batch_size, self.num_attention_heads, seq_len)  # (batch_size, heads, seq_len)
        alphas = self.dropout_layer(alphas)  # (batch_size, heads, seq_len)

        # Average the attention from different hops
        # (batch_size, heads, seq_len) * (batch_size, seq_len, hidden_size)
        # -> (batch_size, heads, hidden_size)
        # -> (batch_size, hidden_size)
        output = torch.mean(torch.bmm(alphas, batch), dim=1)

        return output, alphas

    def compute_attention_weights(self,
                                  batch: FloatTensor,
                                  mask: FloatTensor) -> FloatTensor:
        """
        Computes the attention weights from applying self-attention to the batch.

        :param batch: A FloatTensor of shape `(sequence_length, batch_size, hidden_size)`.
        :param mask: A FloatTensor of shape `(sequence_length, batch_size)` with 1s for content and 0s for padding.
        :return: A FloatTensor of shape `(batch_size, num_attention_heads, sequence_length)` with attention weights.
        """
        return self.compute_output_and_attention_weights(batch, mask)[1]

    def forward(self,
                batch: FloatTensor,
                mask: FloatTensor) -> FloatTensor:
        """
        Applies self-attention to each sequence in the batch.

        :param batch: A FloatTensor of shape `(sequence_length, batch_size, hidden_size)`.
        :param mask: A FloatTensor of shape `(sequence_length, batch_size)` with 1s for content and 0s for padding.
        :return: A FloatTensor of shape `(batch_size, hidden_size)` with the output of the self-attentive layer.
        """
        return self.compute_output_and_attention_weights(batch, mask)[0]
