from typing import *

import torch
from torch import FloatTensor, LongTensor
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
#from attention import SelfAttentiveLayer
from transformers import *


class BertRetrieval(nn.Module):

    def __init__(self, config: Dict):
        """
        :param config: A dictionary containing the model and training configuration.
        """
        super(BertRetrieval, self).__init__()

        
        self.device = torch.device('cuda') if config['cuda'] else torch.device('cpu')
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.pooling = config['bert_pooling']
        self.config = config 

        model = AutoModel.from_pretrained(config['bert_type'])
        if config['bert_freeze_embedding']:
            model.embeddings.requires_grad = False  
        if config['bert_freeze_encoder']:
            model.encoder.requires_grad = False
        bertsize = model.config.hidden_size
        self.embedder = model.to(self.device)
        self.output_size = bertsize 
     
        # outlayer = nn.Sequential(nn.Linear(bertsize, self.hidden_size),
        #                         nn.ReLU(),
        #                         nn.Dropout(self.dropout),
        #                         # nn.Linear(self.hidden_size, self.hidden_size),
        #                         # nn.ReLU(),
        #                         # nn.Dropout(self.dropout)
        #                         )
        # self.outlayer = outlayer.to(self.device)
        # self.output_size = config['hidden_size']

    def change_device(self, d):
        self.device = d


    def encode(self,
                data: torch.Tensor,
                lengths: LongTensor, 
                token_type_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Uses an RNN and self-attention to encode a batch of sequences of word embeddings.
        :param batch: A FloatTensor of shape `(sequence_length, batch_size, embedding_size)` containing embedded text.
        :param lengths: A LongTensor of shape `(batch_size)` containing the lengths of the sequences, used for masking.
        :return: A FloatTensor of shape `(batch_size, output_size)` containing the encoding for each sequence
        in the batch.
        """
        # print(data.shape)
        # Create mask for padding
        max_len = lengths.max().item()
        attention_mask  = torch.zeros(len(data), max_len, dtype=torch.float)
        for i in range(len(data)):
            attention_mask[i, :lengths[i]] = 1

        # if attention_mask is None and self.padding_idx is not None:
        #     attention_mask = (data != self.padding_idx).float()

        data = data.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.embedder(data, attention_mask=attention_mask)
        if self.pooling=='first':
            output = outputs[1]
        elif self.pooling=='average':
            output = outputs[0]
            output = output*attention_mask.unsqueeze(-1)
            output = output.mean(dim=1)
            
        if self.config['bert_freeze_all']:
            # print('freezing all bert weight')
            output = output.detach()
        # output = self.outlayer(output)
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
