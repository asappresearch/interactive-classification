import argparse


from typing import Dict, Iterator, Tuple, Union

from typing import *


from asapp.ml_common.embedders import FastTextEmbedder
from asapp.ml_common.embedders import IndexBatchEmbedder, WordBatchEmbedder 
from asapp.ml_common.interfaces import  Embedder, Preprocessor
from tqdm import tqdm, trange
import os

from collections import defaultdict
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import scipy.stats
import pdb
import time
import math
from torch.distributions.categorical import Categorical
from tensorboardX import SummaryWriter
from torch.nn.functional import normalize

from module.preprocessor import Tokenizer_nltk
# import module.utils as utils

from bdmodule.bd_utils import load_checkpoint, load_model, encode_candidates
from  bdmodule.BUDloader import read_data, split_data
from bdmodule.BUDmodel import Retrieval



class BudAgent(nn.Module):
 
    def __init__(self, args: Dict):
        super(BudAgent, self).__init__()
        self.args = args
        self.budagent_setting()
        #self.budagent_init()

    def budagent_setting(self):
        print('\nInitializing agent setting')
        #=========  loading model and fastext =============
        self.embeddertype = self.args['embeddertype']
        self.batch_size = self.args['batch_size']
        self.device = torch.device('cuda') if self.args['cuda'] else torch.device('cpu')
        self.policy = self.args['strategy']

        self.preprocessor = Tokenizer_nltk()
        print('Loading FastText')
        self.embedder = FastTextEmbedder(path= self.args['embedding_path'] )
        print('loading model from {}'.format(self.args['ckpt']))
        model_state, _, _, _, _, savedconfig = load_checkpoint(self.args['ckpt'])
        savedconfig['cuda'] = self.args['cuda']
        model = Retrieval(savedconfig)
        model.load_state_dict(model_state)

        #model  = load_checkpoint(torch.load( self.args, map_location=lambda storage, location: storage))
        self.model = model.to(self.device)

        self.position = 0
        self.seed = 0


    def agent_init(self):
        self.initialize_data()
        self.initialize_model()


    def initialize_data(self):
        print('\nInitializing agent data')
        #==========loading data =============
        data_records , clsatt = read_data()
        
        self.bdtag_table, self.features, self.clname = self.parse_tag_bd_table()
        data_train, data_val = split_data(data_records, r =0.2)
        self.word_to_index = self.get_vocab(data_records) 

        #self.iqs_eval = self.parse_data(data_val)
        self.iqs_eval, self.tgt_eval = self.parse_data(data_val)
        self.iqs_train, self.tgt_train = self.parse_data(data_train)
        self.iqs, self.tgt_ids = self.iqs_eval, self.tgt_eval
        print('there are in total {} initial queries'.format(len(self.iqs_eval) + len(self.iqs_train)))




    def initialize_model(self):  
        print('\nInitializing agent model')
        #=========  loading model and fastext =============
        print('Loading embeddings')
        if self.args['embeddertype'] == 'index':
            self.batch_embedder = IndexBatchEmbedder(self.embedder, self.word_to_index, cuda=self.args['cuda'])
        elif self.args['embeddertype'] == 'word':
            self.batch_embedder = WordBatchEmbedder(self.embedder, set(self.word_to_index.keys()), cuda=self.args['cuda'])
        else:
            print('bath embedder method not implemented')
        print('\nVocabulary size = {:,}'.format(len(self.word_to_index)))

        self.clname_index = self._preprocess(self.clname)
        self.clname_mat = self.encode_candidates( self.clname_index)

        #self.faqtag_belief = torch.Tensor(self.bdtag_table).to(self.device)
        #=========  set up tag inference module=============
        
        ebed_w_ = 0.8 if self.args['ft_rnn'] else 1.0
        self.embedweight = nn.Parameter(torch.Tensor([ebed_w_]).to(self.device))

        if self.args['taginfer']:
            self.faqtag_belief0 = self.tag_input()
            self.faqtag_belief = self.faqtag_belief0.sigmoid()
            if self.args['ft_tag'] or self.args['tag_pretrain']:
                if self.args['tag_model'] == 'scalar':
                    w_ = np.array([0.4747236]) #np.random.rand(1)
                    b_ = np.array([ -26.986095]) #np.random.rand(1)
                    w_ = np.array([0.312]) #np.random.rand(1)
                    b_ = np.array([1.0]) #np.random.rand(1)
                # elif  self.args['tag_model'] == 'vector':
                #     w_ = np.load('w_813_813_linear.npy')
                #     b_ = np.load('b_813_linear.npy')
                # elif self.args['tag_model'] == 'bl':
                #     nd = self.model.output_size
                #     print(nd)
                #     w_ = np.zeros((nd,nd)) #np.random.rand(1)
                #     b_ = np.array([0.4]) #np.random.rand(1)
                # else:
                #     print('error!!')
                if not self.args['tag_pretrain']:
                    w_ = np.ones(w_.shape)*0.1
                    b_ = np.ones(b_.shape)*0.1
                self.tagweight = nn.Parameter(torch.Tensor(w_).to(self.device))
                self.tagbias = nn.Parameter(torch.Tensor(b_).to(self.device))

                ld_ = self.args['aa_ld'] #0.5 #np.random.rand(1)
                self.lmda = n(torch.Tensor([ld_]).to(self.device))
                self.tag_inference()
        else:
            self.faqtag_belief = torch.Tensor(self.bdtag_table).to(self.device) 










    ## ========================== data processing ==========================

    def parse_tag_bd_table(self):
        home = str(Path.home())
        attnames = open( '../CUB_data/attributes/mod_att.txt').readlines()
        features = []
        for att in attnames:
            tagkey = att.split(' ')[0]
            '''
            astr = att.split(' ')[-1].strip()
            cat = astr.split("::")[0].split('_')[1:]
            tag =  ' '.join(cat + ['is ']) + astr.split("::")[-1]
            tag=tag.lower()
            '''
            tag = ' '.join(att.split(' ')[1:]).strip()
            features.append(tag)

        imgnames = open( '../CUB_data/attributes/images.txt').readlines()
        clsname = []
        for img in imgnames:
            imgkey = img.split(' ')[0]
            imgname = img.split(' ')[-1].strip().split('/')[-1].replace('.jpg','')
            clss  = ' '.join(imgname.lower().split('_')[:-2])
            if clss not in clsname:
                clsname.append(clss)

        scale=2
        clsatt = open( '../CUB_data/attributes/class_attribute_labels_continuous.txt').readlines()
        clsatt=[[float(nm) for nm in x.split(' ')] for x in clsatt]
        clsatt = np.array(clsatt)
        # clstag_table=np.zeros_like(clsatt)
        # r = 50  #clsatt.mean()*scale
        # clstag_table[np.where(clsatt>r)] = 1
        clstag_table = clsatt

        self.tag_i2w = {k: v for k,v in enumerate(features)}
        return clstag_table, features, clsname


    def parse_data(self, data_records):
        # got fields of ['clsname', 'imageid', 'querylist', 'taglist']
        iqs =[]
        tgt_ids=[]

        for i in range(len(data_records)):
            dr =data_records[i]
            iqs += dr['querylist']
            tgtid = self.clname.index(dr['clsname'])
            tgt_ids += [tgtid]*len(dr['querylist'])
        #self.iqs_text = iqs
        iqs = self._preprocess(iqs)
        c = list(zip(iqs, tgt_ids))
        random.shuffle(c)
        iqs, tgt_ids = zip(*c)
        return np.array(iqs), np.array(tgt_ids)


    def get_vocab(self, data_records):
        # got fields of ['clsname', 'imageid', 'querylist', 'taglist']
        w2i = {}
        texts=[]
        for dr  in data_records:
            texts += dr['querylist']
        corpus = self.features + self.clname + texts
        for seq in corpus:
            for w in self.preprocessor.process(seq): 
                if w not in w2i:
                    w2i[w] = len(w2i)
        return w2i


    
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

    def __iter__(self):
        return self


    def __next__(self):
        position, batch_size = self.position, self.batch_size

        if position < len(self.iqs):
            # Get queries and positive targets
            queries = self.iqs[position:position + batch_size]
            targets = self.tgt_ids[position:position + batch_size]
            qr_fact = self.bdtag_table[targets]
            
            # Advance position
            self.position += self.batch_size
            print('positon: {}'.format(self.position))
            return queries, qr_fact, torch.Tensor(targets).long().to(self.device)
        else:
            self.position = 0
            raise StopIteration()


    def sampletrain(self, bs):
        indices = np.random.choice( len(self.iqs_train), bs)
        queries = self.iqs_train[indices]
        targets = self.tgt_train[indices]
        qr_fact = self.bdtag_table[targets]
        return queries, qr_fact, torch.Tensor(targets).long().to(self.device)


    def sampleeval(self, bs):
        indices = np.random.choice( len(self.iqs_eval), bs)
        queries = self.iqs_eval[indices]
        targets = self.tgt_eval[indices]
        qr_fact = self.bdtag_table[targets]
        return queries, qr_fact, torch.Tensor(targets).long().to(self.device)

    def valdata(self):
        queries = self.iqs_eval  
        targets = self.tgt_eval
        qr_fact = self.bdtag_table[targets]
        return queries, qr_fact, torch.Tensor(targets).long().to(self.device)




    ## ========================== Encoding, information gain etc ==========================
    def encode_candidates(self, data):
        cand_mat = []
        batch_size = self.batch_size
        num_batch = len(data) // batch_size

        for batch_idx in range(num_batch + 1):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size
            batch = data[start_idx:end_idx]
            if len(batch) == 0:
                break
            cand_embeddings, cand_lengths = self.batch_embedder.embed(batch)
            cand_encodings = self.model.encode_class(cand_embeddings, cand_lengths)
            cand_mat.extend(cand_encodings)
        return torch.stack(cand_mat).detach()


    def rankbatch(self, queries):
        self.faqs_mat = encode_candidates(self.clname_index, self.batch_embedder, self.model, self.batch_size)#self.encode_candidates( self.faqs_index)
        qr_embeddings, qr_lengths = self.batch_embedder.embed(queries)
        query_encodings = self.model.encode_context(qr_embeddings, qr_lengths)
        score = query_encodings @ self.faqs_mat.t()
        if not self.args['ft_rnn']:
            score = score.detach()
        score = score * self.embedweight
        score = F.softmax(score, -1)  ### Why the softmax actually make a difference 
        return score




    def infogain_batch(self, score, ft_asked=None, debug = False):
        with torch.no_grad():
            p_f_x = score
            p_a_f = self.faqtag_belief.t()
            pos_entropy = conditional_entropy(p_f_x, p_a_f)
            neg_entropy = conditional_entropy(p_f_x, 1-p_a_f)
            weight_entropy = pos_entropy + neg_entropy  # n_bs * n_totalft
            #if ft_asked !=None :
            # a littel hacky solution to repeated asked questions
            if ft_asked and self.args['no_rpt_ft']:
                mask = torch.zeros( weight_entropy.shape).to(self.device)
                # mask[:,ft_asked] = 10
                for i in range(weight_entropy.shape[0]):
                    mask[i,ft_asked[i]] = 10
                weight_entropy += mask #torch.Tensor(mask).to(self.device)
            best_f = weight_entropy.argmin(dim=1)  # information gain = total_entropy- weighted_entropy   
            return best_f



def conditional_entropy(p_f_x, p_a_f):
    paf_pfx = p_f_x.unsqueeze(1) * p_a_f 
    p_f_ax = (1e-12 + paf_pfx )/torch.sum(paf_pfx + 1e-12 , 2 , keepdim=True)
    #p_f_ax =  normalize(paf_pfx, p=1, dim=-1)
    new_entropy = Categorical(p_f_ax).entropy() 
    p_a_x = torch.sum(paf_pfx, 2)  
    weighted_entropy = new_entropy * p_a_x
    return weighted_entropy