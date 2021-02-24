from typing import *

from sru import SRU
import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from utils.helpers import create_mask
from  module.attention import SelfAttentiveLayer

class FAQRetrieval(nn.Module):

    def __init__(self, config: Dict):
        """
        :param config: A dictionary containing the model and training configuration.
        """
        super(FAQRetrieval, self).__init__()

        
        self.device = torch.device('cuda') if config['cuda'] else torch.device('cpu')

        self.embedding_size = config['embedding_size']
        self.hidden_size = config['hidden_size']
        self.bidirectional = config['bidirectional']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.rnn_dropout = config['rnn_dropout']
        self.output_size = self.hidden_size * (1 + self.bidirectional)

        self.dropout_layer = nn.Dropout(self.dropout)
        self.rnn = SRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            rnn_dropout=self.rnn_dropout,
            bidirectional=self.bidirectional,
            use_tanh=False,
            layer_norm=False,
            rescale=False
        )
        self.config = config
        if self.config['use_attention']:
           self.attention = SelfAttentiveLayer(self.output_size, config)
        self.candi_mat = None
    
    def change_device(self, d):
        self.device = d


    def encode(self,
               batch: FloatTensor,
               lengths: LongTensor) -> FloatTensor:
        """
        Uses an RNN and self-attention to encode a batch of sequences of word embeddings.

        :param batch: A FloatTensor of shape `(sequence_length, batch_size, embedding_size)` containing embedded text.
        :param lengths: A LongTensor of shape `(batch_size)` containing the lengths of the sequences, used for masking.
        :return: A FloatTensor of shape `(batch_size, output_size)` containing the encoding for each sequence
        in the batch.
        """
        # Create mask for padding
        mask = create_mask(lengths, cuda=self.device == torch.device('cuda') ) #next(self.parameters()).is_cuda)

        # Dropout
        batch = self.dropout_layer(batch)

        # Sequence encoding
        output, _ = self.rnn(batch, mask_pad=(1 - mask.float()))

        # Dropout
        output = self.dropout_layer(output)

        if self.config['use_attention']:
           output = self.attention(output, mask=mask.float())
        else:
            # Get the last 
            lengths = lengths.type(torch.LongTensor).to(self.device)
            #if self.usecuda:
            #lengths = lengths.cuda()
            output = output[lengths - 1, np.arange(len(lengths)),:] 

        return output


    def encode_context(self,
                       context: FloatTensor,
                       lengths: LongTensor) -> FloatTensor:
        """
        Encode a batch of embedded contexts.

        :param context: A FloatTensor of shape `(sequence_length, batch_size, embedding_size)` containing embedded
        contexts (sequences of word embeddings padded to same length).
        :param lengths: A LongTensor of shape `(batch_size)` containing the lengths of the sequences, used for masking.
        :return: A FloatTensor of shape `(batch_size, hidden_size)` of context encoding vectors.
        """
        return self.encode(context, lengths)



    def encode_class(self,
                        response: FloatTensor,
                        lengths: LongTensor) -> FloatTensor:
        """
        Encode a batch of embedded responses.

        :param response: A FloatTensor of shape (sequence_length, batch_size, embedding_size)` of embedded
        responses (sequences of word embeddings padded to same length).
        :param lengths: A LongTensor of shape `(batch_size)` containing the lengths of the sequences, used for masking.
        :return: A FloatTensor of shape `(batch_size, hidden_size)` of response encoding vectors.
        """
        return self.encode(response, lengths)

    @staticmethod
    def score(encoding_1: FloatTensor,
              encoding_2: FloatTensor,
              scoring_method: str) -> FloatTensor:
        """
        Scores a batch of vector pairs using either cosine similarity or dot product.

        :param encoding_1: A FloatTensor of shape `(batch_size, hidden_size)`.
        :param encoding_2: A FloatTensor of shape `(batch_size, hidden_size)`.
        :param scoring_method: The method of scoring the pairs of encodings
        (choices = ['dot product', 'cosine similarity'])
        :return: A FloatTensor of shape `(batch_size)` containing the score for each pair of vectors.
        """
        if scoring_method == 'cosine similarity':
            return F.cosine_similarity(encoding_1, encoding_2, dim=-1)
        elif scoring_method == 'dot product':
            # (batch_size, 1, hidden_dim) x (batch_size, hidden_dim, 1)
            return torch.bmm(encoding_1.unsqueeze(1), encoding_2.unsqueeze(2)).view(-1)
        else:
            raise ValueError('scoring method "{}" not supported'.format(scoring_method))



    def compute_loss(self, output, target):
        batch_size = target.size(0) // (1 + self.num_neg)  # 1 positive and num_neg negatives per batch

        assert (target[:batch_size].data == 1).all()
        assert (target[batch_size:].data == 0).all()

        if self.loss_type == 'hinge':
            positive = output[:batch_size].repeat(self.num_neg)
            negative = output[batch_size:]

            hinge_loss = (negative + self.hinge_margin - positive).clamp(min=0.0).mean()

            return hinge_loss
        elif self.loss_type == 'cross_entropy':
            # Get positive and all negatives for each src on a row
            # [[pos_1, neg_11, neg_12, neg_13, ...],
            #  [pos_2, neg_21, neg_22, neg_23, ...],
            #   ...
            #  [pos_n, neg_n1, neg_n2, neg_n3, ...]]
            output = output.reshape(self.num_neg + 1, batch_size).t()
            target = torch.zeros(batch_size).long()  # zeros as target b/c positive always in 0th index
            #target = target.cuda() if self.use_cuda else target
            target = target.to(self.device)
            ce_loss = F.cross_entropy(output, target)

            return ce_loss
        elif self.loss_type == 'binary_cross_entropy':
            bce_loss = F.binary_cross_entropy_with_logits(output, target.float())

            return bce_loss
        elif self.loss_type == 'weighted_bce':
            target = target.float()

            output_pos, output_neg = output[:batch_size], output[batch_size:]
            target_pos, target_neg = target[:batch_size], target[batch_size:]

            bce_loss_pos = F.binary_cross_entropy_with_logits(output_pos ,target_pos)
            bce_loss_neg = F.binary_cross_entropy_with_logits(output_neg, target_neg)

            weighted_bce_loss = self.num_neg * bce_loss_pos + bce_loss_neg

            return weighted_bce_loss
        else:
            raise ValueError('Loss type "{}" not supported.'.format(self.loss_type))




    # def predict(self, queries, candidates):
    #     if self.candi_mat is None:
    #         self.candi_mat = utils.encode_candidates(candidates, self.batch_embedder, self.model, self.batch_size)#self.encode_candidates( self.faqs_index)

    #     score = self.forward(queries, candidates)
    #     prob = F.softmax(score, -1)  ### Why the softmax actually make a difference 
    #     return prob

    # def encode_candidates(self, queries, candidates):
    #     if self.candi_mat is None:
    #         self.candi_mat = utils.encode_candidates(candidates, self.batch_embedder, self.model, self.batch_size)#self.encode_candidates( self.faqs_index)

    #     score = self.forward(queries, candidates)
    #     prob = F.softmax(score, -1)  ### Why the softmax actually make a difference 
    #     return prob