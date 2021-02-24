import argparse

import yaml
from typing import Dict, Iterator, Tuple, Union

from typing import *
import numpy as np
import pandas as pd
import random 

from asapp.ml_common.embedders import FastTextEmbedder
from asapp.ml_common.embedders import IndexBatchEmbedder, WordBatchEmbedder 
from asapp.ml_common.interfaces import  Embedder
from tqdm import tqdm, trange
import os

from collections import defaultdict
import argparse

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

from  bdmodule.BUDloader import read_data, split_data
import bdmodule.bd_utils as bd_utils
import utils.utils as utils
from utils.helpers import load_checkpoint
# from bdmodule.BUDmodel import Retrieval

# from bdmodule.BudAgent import BudAgent

from bdmodule.BUDmodel import Retrieval


class BudnoiseAgent(nn.Module):


    def __init__(self, args: Dict):
        super(BudnoiseAgent, self).__init__()
        # BudAgent.__init__(self, args)
        self.args = args
        print('\nInitializing agent setting')
        #=========  loading model and fastext =============
        self.embeddertype = self.args['embeddertype']
        self.batch_size = self.args['batch_size']
        self.device = torch.device('cuda') if self.args['cuda'] else torch.device('cpu')
        self.policy = self.args['strategy']

        self.position = 0
        self.seed = 0

        if args['bert']:
            from data.berttokenizer import BTTokenizer, BertBatcher
            from module.bert_trainer import run_epoch
            from module.bertmodel import BertRetrieval as Retrieval
            from transformers import AdamW, WarmupLinearSchedule
            print('loading bert tokenizer')
            self.preprocessor = BTTokenizer(args) #SplitPreprocessor()
            PAD_ID = self.preprocessor.padding_idx()
            self.batch_embedder = BertBatcher(cuda=args['cuda'], pad=PAD_ID)
            self.embeddertype = 'bpe'
        else:
            from data.tokenizer import Tokenizer_nltk
            from module.trainer import run_epoch
            from bdmodule.BUDmodel import Retrieval
            self.preprocessor = Tokenizer_nltk() #SplitPreprocessor()


        print('\n\nloading model from {}'.format(args['ckpt']))
        if self.args['ckpt'].endswith('newtk_pretrianed.pt'): 
            state = torch.load(args['ckpt'], map_location=lambda storage, location: storage)
            model = state['model'].eval()
        else:
            model_state, _, _, _, _, savedconfig = load_checkpoint(self.args['ckpt'])
            savedconfig['cuda'] = self.args['cuda']
            model = Retrieval(savedconfig)
            model.load_state_dict(model_state)
        self.model = model.to(self.device)
        

    def agent_init(self):
        self.initialize_data()
        self.initialize_model()

    def initialize_data(self):
        print('\nInitializing agent data')
        #==========loading data =============
        data_records , clsatt = read_data(self.args)
        
        self.clname = list(pd.DataFrame.from_records(data_records).clsname.unique())

        data_train, data_val = split_data(data_records, r =0.2)

        self.bdtag_table, self.features = self.parse_tag_bd_table(data_records )
        print('total clsname: {}'.format(len(self.clname)))

        self.bdtag_dev, _= self.parse_tag_bd_table(data_val )

        self.word_to_index = self.get_vocab(data_records) 

        #self.val_queries, self.val_targets = parse_valdata(data_val, clsatt, 1, 1 , includetag=False, sampleN=5)
        self.iqs_eval, self.tgt_eval, self.attr_eval  = self.parse_data(data_val)
        self.iqs_train, self.tgt_train, self.attr_train = self.parse_data(data_train)
        self.iqs, self.tgt_ids, self.attr = self.iqs_eval, self.tgt_eval, self.attr_eval

        print('there are in total {} initial queries'.format(len(self.iqs_eval) + len(self.iqs_train)))

    def initialize_model(self):  
        print('\nInitializing agent model')
        #=========  loading model and fastext =============
        if not self.args['bert']:
            self.embedder = FastTextEmbedder(path= self.args['embedding_path'] )
            print('Loading embeddings')
            if self.args['embeddertype'] == 'index':
                self.batch_embedder = IndexBatchEmbedder(self.embedder, self.word_to_index, cuda=self.args['cuda'])
            elif self.args['embeddertype'] == 'word':
                self.batch_embedder = WordBatchEmbedder(self.embedder, set(self.word_to_index.keys()), cuda=self.args['cuda'])
            else:
                print('bath embedder method not implemented')
            print('\nVocabulary size = {:,}'.format(len(self.word_to_index)))

        self.clname_index = self._preprocess(self.clname)
        # self.clname_mat = self.encode_candidates( self.clname_index)
        with torch.no_grad():
            self.clname_mat = utils.encode_candidates(self.clname_index, self.batch_embedder, self.model, self.batch_size)#self.encode_candidates( self.faqs_index)

        #self.faqtag_belief = torch.Tensor(self.bdtag_table).to(self.device)
        #=========  set up tag inference module=============
        self.embedweight = nn.Parameter(torch.Tensor([0.8]).to(self.device))
        if self.args['taginfer']:
            self.faqtag_belief0 = self.tag_input()
            self.faqtag_belief = self.faqtag_belief0.sigmoid()
            if self.args['ft_tag'] or self.args['tag_pretrain']:
                if self.args['tag_model'] == 'scalar':
                    w_ = np.array([0.0]) #np.random.rand(1)
                    b_ = np.array([0.0]) #np.random.rand(1)
                    ld_ = np.array([0.0])
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
                ld_ = self.args['aa_ld']
                self.tagweight = nn.Parameter(torch.Tensor(w_).to(self.device))
                self.tagbias = nn.Parameter(torch.Tensor(b_).to(self.device))
                self.lmda = nn.Parameter(torch.Tensor([ld_]).to(self.device))
                self.tag_inference()
        else:
            self.faqtag_belief = torch.Tensor(self.bdtag_table).to(self.device) 


    def _preprocess(self, texts: List[str]) -> List[List[int]]:
        """
        Preprocesses a list of strings by applying the preprocessor/tokenizer, truncating, and mapping to word indices.
        """
        indices = []
        for text in texts:
            word_sequence = self.preprocessor.process(text)  # preprocess/tokenize and truncate
            if self.embeddertype == 'word' or self.embeddertype=='bpe':
                indices.append(word_sequence)
                continue
            index_sequence = []
            for word in word_sequence:
                index_sequence.append(self.word_to_index.setdefault(word, len(self.word_to_index)))
            if self.embeddertype == 'index':
                indices.append(index_sequence)
        return indices

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

    ## ========================== data processing ==========================
    def parse_tag_bd_table(self, data_records):

        tag_w2i, tag_i2w = bd_utils.get_att_dict()
        label2att, att2label = bd_utils.att2label()
        self.nq_bi = len(tag_w2i) 
        if self.args['using_categorical']:
            binary_idx, categorical, tag_new2old, tag_old2new = bd_utils.parse_cat_binary()
            self.tag_w2i ={w:tag_old2new[i] for w,i in tag_w2i.items()}
            self.tag_i2w = {v:k for k,v in self.tag_w2i.items()}
            self.label2att = {lb:[tag_old2new[i] for i in label2att[lb]] for lb in label2att}
            self.att2label = [att2label[tag_new2old[new]]  for new in tag_new2old]
            self.categorical = categorical
            self.nq_bi = len(binary_idx)
            self.catq = [cat['question'] for cat in categorical]
            print(self.categorical)
        else:
            self.tag_w2i, self.tag_i2w = tag_w2i, tag_i2w
            self.label2att, self.att2label = label2att, att2label

        features = list(self.tag_w2i.keys())

        # cls_att = yaml.load(open('cls_att_average.yml', 'r'))
        # clsname = []
        # clsatt = []
        # for cls,vec in cls_att.items():
        #     clsname.append(cls)
        #     clsatt.append(vec)
            
        def taglist_tovec(taglist):
                #taglist = img.taglist
                vec = np.zeros(len(self.tag_w2i))
                tgids = [self.tag_w2i[t] for t in taglist]
                vec[np.array(tgids)] = 1
                return vec

        #clsname = []
        #### Compose a class level believe
        clsatt = np.ones((200, len(features)))/2
        alldata = pd.DataFrame.from_records(data_records)
        for cls,examples in alldata.groupby(['clsname']):
            #clsname.append(cls)
            test = examples
            test['vector'] = test.apply(lambda x: taglist_tovec(x.taglist), axis=1)

            label_vecs= np.zeros(len(self.label2att))
            for vec in test.vector:
                labelvec = [int(vec[self.label2att[label]].sum()!=0) for label in self.label2att.keys()]
                label_vecs += np.array(labelvec)
            att_count=[label_vecs[x] for x in self.att2label]
            ave_vec = test.vector.sum()/np.array(att_count)
            clsatt[ self.clname.index(cls)] = ave_vec

        # if self.args['tag_threshold']!=0:  # Default as 0
        #     thresh = self.args['tag_threshold']
        #     print(f'Using threhold {thresh} to clean p(a,d) belief')
        #     super_threshold_indices = clsatt < thresh
        #     print(super_threshold_indices)
        #     clsatt[super_threshold_indices] = 0

        if self.args['using_categorical']:
            for cat in self.categorical:
                clsatt[:, cat['idx']] = clsatt[:, cat['idx']]/clsatt[:, cat['idx']].sum(1, keepdims=True)
           
        clstag_table = clsatt
        # self.tag_i2w = {k: v for k,v in enumerate(features)}
        return clstag_table,   features


    def parse_data(self, data_records):
        # got fields of ['clsname', 'imageid', 'querylist', 'taglist']
        iqs =[]
        tgt_ids=[]
        attrs = []

        for i in range(len(data_records)):
            dr =data_records[i]
            iqs += dr['querylist']
            tgtid = self.clname.index(dr['clsname'])
            tgt_ids += [tgtid]*len(dr['querylist'])
            att =[self.tag_w2i[tg] for tg in dr['taglist']]
            for i in range(len(dr['querylist'])):
                attrs.append(att)
        iqs = self._preprocess(iqs)
        c = list(zip(iqs, tgt_ids, attrs))
        random.shuffle(c)
        iqs, tgt_ids, attrs = zip(*c)
        attrs= np.array(attrs)

        # print(attrs[2])
        # print(attrs.shape)
        # print( self.bdtag_table.shape)
        # if self.args['using_categorical']:
        #     for cat in self.categorical:
        #         attrs[:, cat['idx']] = attrs[:, cat['idx']]/(1e-2 + attrs[:, cat['idx']].sum(1, keepdims=True))

        return np.array(iqs), np.array(tgt_ids), attrs


    def __iter__(self):
        return self


    def __next__(self):
        position, batch_size = self.position, self.batch_size

        if position < len(self.iqs):
            # Get queries and positive targets
            queries = self.iqs[position:position + batch_size]
            targets = self.tgt_ids[position:position + batch_size]
            qr_fact = self.attr[position:position + batch_size]
            
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
        #qr_fact = self.bdtag_table[targets]
        qr_fact = self.attr_train[indices]
        return queries, qr_fact, torch.Tensor(targets).long().to(self.device)


    def sampleeval(self, bs):
        indices = np.random.choice( len(self.iqs_eval), bs)
        queries = self.iqs_eval[indices]
        targets = self.tgt_eval[indices]
        qr_fact = self.attr_eval[indices]
        return queries, qr_fact, torch.Tensor(targets).long().to(self.device)

    def valdata(self):
        queries = self.iqs_eval  
        targets = self.tgt_eval
        qr_fact = self.bdtag_table[targets]
        return queries, qr_fact, torch.Tensor(targets).long().to(self.device)



    # ## ========================== Encoding, information gain etc ==========================
    def tag_input(self):
        '''
        print('\n\nloading pretrained tag model from {}'.format(self.self.args['tagckpt']))
        tagmodel  = load_checkpoint(self.self.args['tagckpt'])[0]
        self.tagmodel = tagmodel.to(self.device)
        '''
        self.tagmodel = self.model
        ft_index = self._preprocess(self.features)
        ft_embeddings, ft_lengths = self.batch_embedder.embed(ft_index)
        ft_encodings = self.tagmodel.encode_context(ft_embeddings, ft_lengths).detach()

        faq_embeddings, faq_lengths = self.batch_embedder.embed(self.clname_index)
        faq_encodings = self.tagmodel.encode_class(faq_embeddings, faq_lengths).detach()

        self.faq_encodings = faq_encodings
        self.ft_encodings = ft_encodings

        scores = faq_encodings @ft_encodings.t()

        assert scores.shape ==  self.bdtag_table.shape
        faqtag_belief0 = scores.data
        return faqtag_belief0


    def tag_inference(self):
        if self.args['tag_model'] == 'scalar':
            pafraw = self.faqtag_belief0* self.tagweight + self.tagbias

        # if self.self.args['tag_model'] == 'vector':
        #     paf = self.faqtag_belief0 @ self.tagweight + self.tagbias.unsqueeze(0)

        # if self.self.args['tag_model'] == 'bl':
        #     paf = (self.faq_encodings @ self.tagweight) @ self.ft_encodings.t() + self.tagbias
        #     paf = paf.squeeze(0)

        # paf = paf.sigmoid()
        if self.args['using_categorical']:
            paf_binary = pafraw[:, :self.nq_bi].sigmoid()
            paf_cat = []
            for cat in self.categorical:
                pafcat = pafraw[:, cat['idx']].softmax(dim=1 )   ## Need to fix the index
                paf_cat.append(pafcat)
            paf = torch.cat( (paf_binary, *paf_cat), 1)
        else:
            paf = pafraw.sigmoid()
            
        belief_t = self.lmda * paf + (1- self.lmda)* torch.Tensor(self.bdtag_table).to(self.device)
        self.faqtag_belief  = belief_t



    def infogain_batch(self, score, ft_asked=None, debug = False):
        with torch.no_grad():
            p_f_x = score
            # total_entropy = compute_entropy(p_f_x)

            # compute IG of binary questions
            p_a_f = torch.tensor(self.faqtag_belief[:, :self.nq_bi], dtype=torch.float).to(self.device).t()#self.faqtag_belief.t()
            # p_a_f = self.faqtag_belief.t()
            pos_entropy = conditional_entropy(p_f_x, p_a_f)
            neg_entropy = conditional_entropy(p_f_x, 1-p_a_f)
            b_weight_entropy = pos_entropy + neg_entropy   # batch * n_binary

            if self.args['using_categorical']:
                cat_entropy = []
                for cat in self.categorical:
                    entropy = 0
                    for idx in cat['idx']:
                        p_a_f = torch.tensor(self.faqtag_belief[:, idx], dtype=torch.float).unsqueeze(1).to(self.device).t()
                        # print(p_a_f.shape)
                        entropy += conditional_entropy(p_f_x, p_a_f)
                    cat_entropy.append(entropy)

                weight_entropy = torch.cat((b_weight_entropy, *cat_entropy ), 1 )
            else:
                weight_entropy = b_weight_entropy

            if ft_asked and self.args['no_rpt_ft']:
                mask = torch.zeros( weight_entropy.shape).to(self.device)
                # mask[:,ft_asked] = 10
                print(ft_asked)
                for i in range(weight_entropy.shape[0]):
                    print(ft_asked[i])
                    mask[i,ft_asked[i]] = 100
                weight_entropy += mask #torch.Tensor(mask).to(self.device)

            best_q = weight_entropy.argmin(dim=1)  # information gain = total_entropy- weighted_entropy
            return best_q #self.reform_ft(best_f)


    # def infogain_batch_binary(self, score, ft_asked=None, debug = False):
    #     with torch.no_grad():
    #         p_f_x = score
    #         p_a_f = self.faqtag_belief.t()
    #         pos_entropy = conditional_entropy(p_f_x, p_a_f)
    #         neg_entropy = conditional_entropy(p_f_x, 1-p_a_f)
    #         weight_entropy = pos_entropy + neg_entropy  # n_bs * n_totalft
    #         #if ft_asked !=None :
    #         # a littel hacky solution to repeated asked questions
    #         if ft_asked and self.args['no_rpt_ft']:
    #             mask = torch.zeros( weight_entropy.shape).to(self.device)
    #             # mask[:,ft_asked] = 10
    #             for i in range(weight_entropy.shape[0]):
    #                 mask[i,ft_asked[i]] = 10
    #             weight_entropy += mask #torch.Tensor(mask).to(self.device)
    #         best_f = weight_entropy.argmin(dim=1)  # information gain = total_entropy- weighted_entropy   
    #         return best_f

    def question_generator(self, p_f_x, ft_asked):
        best_ft = self.infogain_batch(p_f_x, [ft_asked])[0]
        print(f'best ft : {best_ft}')
        if best_ft < self.nq_bi:
            best_feature = self.tag_i2w[best_ft.cpu().item()]
            question = f'Does the bird have {best_feature} ?'
            answers =  ['yes', 'no']
        else:
            cat = self.categorical[ best_ft - self.nq_bi]
            question = 'What is the bird' + cat['question'].replace('has', '').replace('_',' ')+'?'
            answers = [a.replace('_',' ') for a in cat['answers']]
        return question, answers, [best_ft]


    def rankbatch(self, queries):
        # self.faqs_mat = encode_candidates(self.clname_index, self.batch_embedder, self.model, self.batch_size)#self.encode_candidates( self.faqs_index)
        qr_embeddings, qr_lengths = self.batch_embedder.embed(queries)
        query_encodings = self.model.encode_context(qr_embeddings, qr_lengths)

        score = query_encodings @ self.clname_mat.t()
        if not self.args['ft_rnn']:
            score = score.detach()
        score = score * self.embedweight
        score = F.softmax(score, -1)  ### Why the softmax actually make a difference 
        return score



def conditional_entropy(p_f_x, p_a_f):
    paf_pfx = p_f_x.unsqueeze(1) * p_a_f 
    p_f_ax = (1e-12 + paf_pfx )/torch.sum(paf_pfx + 1e-12 , 2 , keepdim=True)
    #p_f_ax =  normalize(paf_pfx, p=1, dim=-1)
    new_entropy = Categorical(p_f_ax).entropy() 
    p_a_x = torch.sum(paf_pfx, 2)  
    weighted_entropy = new_entropy * p_a_x
    return weighted_entropy