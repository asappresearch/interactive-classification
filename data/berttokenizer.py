import torch
from transformers import *
from typing import List
from torch.nn.utils.rnn import pad_sequence

class BTTokenizer():    
    """
    Preprocessor that splits text on whitespace into tokens.
    Example:
        >>> preprocessor = SplitPreprocessor()
        >>> preprocessor.process('Hi how may I help you?')
        ['Hi', 'how', 'may', 'I', 'help', you?']
    """
    def __init__(self, config):
        self.tokenizer = AutoTokenizer.from_pretrained(config['bert_type']) 
        self.add_special_tokens = True
        self.max_len = config['max_text_length']

    def padding_idx(self) -> int:
        """Get the padding index.

        Returns
        -------
        int
            The padding index in the vocabulary

        """
        # pad_token = tokenizer._convert_token_to_id(tokenizer.pad_token)
        pad_token = self.tokenizer.pad_token
        return self.tokenizer.convert_tokens_to_ids(pad_token)

    def process(self, text: str) -> List[str]:
        """Split text on whitespace into tokens."""
        tokens = self.tokenizer.encode(text, add_special_tokens=self.add_special_tokens, max_length=self.max_len)
        return tokens


class BertBatcher():
    def __init__(self, cuda, pad):
        self.device = torch.device('cuda') if cuda else torch.device('cpu')
        self.pad = pad


    def embed(self, arr, dtype=torch.long):
        pad_token = self.pad
        lens = torch.LongTensor([len(a) for a in arr])
        max_len = lens.max().item()
        padded = torch.ones(len(arr), max_len, dtype=dtype) * pad_token
        mask = torch.zeros(len(arr), max_len, dtype=torch.long)
        for i, a in enumerate(arr):
            padded[i, :lens[i]] = torch.tensor(a, dtype=dtype)
            mask[i, :lens[i]] = 1
        return padded, lens #, mask

    # def embed(self, input):
    #     input = [torch.tensor(x) for x in input]
    #     pad_input = pad_sequence(input, batch_first=True,  padding_value=self.pad).to(self.device)
    #     return pad_input, 0