
import argparse

from typing import Dict, Iterator, Tuple, Union

from typing import *
import json

from data.embedders.fasttext_embedder import FastTextEmbedder
from data.embedders.batch_embedder import IndexBatchEmbedder, WordBatchEmbedder 

from tqdm import tqdm, trange
import os
import sys

from collections import defaultdict
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import scipy.stats
import pdb
import time
import math
from torch.distributions.categorical import Categorical
from torch.nn.functional import normalize


from utils.helpers import compute_grad_norm, compute_param_norm, \
    load_checkpoint, get_params, noam_step, parameter_count, repeat_negatives, save_checkpoint
from utils.config import  _get_parser
import utils.utils as utils

import data.faq_loader as data_loading

from data.tokenizer import Tokenizer_nltk

from module.model import FAQRetrieval
from module.bertmodel import   BertRetrieval


datapath = '../faq_data/'


class AskingAgent( nn.Module ):

    def __init__(self, args: Dict):
        super(AskingAgent, self).__init__()

        self.args = args
        self.embeddertype = args['embeddertype']
        self.batch_size = args['batch_size']
        self.device = torch.device('cuda') if args['cuda'] else torch.device('cpu')
        self.policy = args['strategy']

        #==========loading data =============

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
            from module.model import FAQRetrieval as Retrieval
            self.preprocessor = Tokenizer_nltk() #SplitPreprocessor()


        #data_records = read_data(args)

        self.queryfile, self.faq_pool= data_loading.read_queries(args)
        data_train, data_val, data_test = data_loading.query_cv(self.queryfile, fold_n= args['cv_n'])
        self.pd_train, self.pd_val, self.pd_test = data_train, data_val, data_test

        self.gold_table, self.tags, self.faqs = self._parse_faq( self.queryfile.to_dict('records') )
        self.faqtag_table = self.reload_fromprob()

        #=========  preprocessing and precompute =============
        self.iqs = np.array(self._preprocess(self.iqs_text))

        # if args['datasplit'] == 'query':
        #     self.iqs_train, self.tgt_train, self.iqs_eval, self.tgt_eval= split_srctgtdata(self.iqs, self.tgt_ids, r = 0.2)
        # elif args['datasplit'] == 'faq':
            #data_train, data_val, data_test = dataloader1129.split_data_tdt(allquery, 0.2, 0.2)
        # data_train, data_val, data_test = data_loading.query_cv(self.queryfile, fold_n= args['cv_n'])
        # self.pd_train, self.pd_val, self.pd_test = data_train, data_val, data_test
        data_train, data_val, data_test  = data_train.to_dict('records'), data_val.to_dict('records'), data_test.to_dict('records') 
        self.iqs_eval, self.tgt_eval = self.get_data(data_val)
        self.iqs_train, self.tgt_train = self.get_data(data_train + data_val)
        self.iqs_test, self.tgt_test = self.get_data(data_test)
        # else:
        #     print('train test splitting error')
        # #self.num_examples = len(self.iqs_eval)
        print('There are {} initial queries from {} faqs to test'.format( len(self.tgt_eval) , len(set(self.tgt_eval)) ))

        self.num_batches = math.ceil( len(self.iqs_train) / self.batch_size)
        self.position = 0

        #=========  loading encoding model and fastext =============
        if not args['bert']:
            print('Loading FastText')
            self.embedder = FastTextEmbedder(path= args['embedding_path'] )
            print('Loading embeddings')
            if args['embeddertype'] == 'index':
                self.batch_embedder = IndexBatchEmbedder(self.embedder, self.word_to_index, cuda=args['cuda'])
            elif args['embeddertype'] == 'word':
                self.batch_embedder = WordBatchEmbedder(self.embedder, set(self.word_to_index.keys()), cuda=args['cuda'])
            else:
                print('bath embedder method not implemented')
            print('\nVocabulary size = {:,}'.format(len(self.word_to_index)))

        print('\n\nloading model from {}'.format(args['ckpt']))
        if self.args['ckpt'].endswith('newtk_pretrianed.pt'): 
            state = torch.load(args['ckpt'], map_location=lambda storage, location: storage)
            model = state['model'].eval()
        else:
            model_state, _, _, _, _, savedconfig = load_checkpoint(self.args['ckpt'])
            savedconfig['cuda'] = self.args['cuda']
            model = Retrieval(savedconfig)
            model.load_state_dict(model_state)
        model.change_device(self.device)
        self.model = model.to(self.device)
        
        self.faqs_index = self._preprocess(self.faqs)
        print('total faqs: {}'.format(len(self.faqs)))
        with torch.no_grad():
            self.faqs_mat = utils.encode_candidates(self.faqs_index, self.batch_embedder, self.model, self.batch_size)#self.encode_candidates( self.faqs_index)

        self.embedweight = nn.Parameter(torch.Tensor([0.8]).to(self.device))

        #=========  set up tag inference module=============
        if args['taginfer']:
            self.faqtag_belief0 = self.tag_input()
            self.faqtag_belief = self.faqtag_belief0.sigmoid()
            if args['ft_tag'] or args['tag_pretrain']:
                if args['tag_model'] == 'scalar':
                    w_ = np.array([0.4747236]) #np.random.rand(1)
                    b_ = np.array([ -26.986095]) #np.random.rand(1)
                    w_ = np.array([0.312]) #np.random.rand(1)
                    b_ = np.array([1.0]) #np.random.rand(1)
                elif  args['tag_model'] == 'vector':
                    w_ = np.load('w_813_813_linear.npy')
                    b_ = np.load('b_813_linear.npy')
                elif args['tag_model'] == 'bl':
                    nd = self.model.output_size
                    print(nd)
                    w_ = np.zeros((nd,nd)) #np.random.rand(1)
                    b_ = np.array([0.4]) #np.random.rand(1)
                else:
                    print('error!!')
                if not args['tag_pretrain']:
                    w_ = np.ones(w_.shape)*0.1
                    b_ = np.ones(b_.shape)*0.1
                self.tagweight = nn.Parameter(torch.Tensor(w_).to(self.device))
                self.tagbias = nn.Parameter(torch.Tensor(b_).to(self.device))

                ld_ = args['aa_ld'] #0.5 #np.random.rand(1)
                self.lmda = nn.Parameter(torch.Tensor([ld_]).to(self.device))
                self.tag_inference()
        else:
            self.faqtag_belief = torch.Tensor(self.faqtag_table).to(self.device) 

        
    ## ========================== data processing ==========================
    def _parse_faq(self, data_records):
        #records has fields of 'faq_id', 'faq_original', 'taglist', 'device', 'querylist'
        tag_w2i = {}
        row, col = [], []
        candidates, iqs, tgt_ids = [], [], []

        self.valtestid = [] 
        self.trainid = []

        for i in range(len(data_records)):
            dr = data_records[i]
            iqs.extend( dr['querylist'])
            tgt_ids += [i] * len(dr['querylist'])
            tgttext = dr['faqtext'] if 'faqtext' in dr else dr['faq_original']
            #tgttext = dr['faq_original']
            candidates.append(tgttext)
            # 'zero shot' for training, don't include questions from devtest set. Filtering the question part
            if (not self.args['zeroshot']) or (self.args['zeroshot'] and dr['faq_id'] in list(self.pd_train.faq_id)):
                for tag in dr['taglist']:
                    tag = ' '.join(tag.strip().split())
                    tagid = tag_w2i.setdefault(tag, len(tag_w2i))
                    row.append(i)
                    col.append(tagid)
            if dr['faq_id'] in list(self.pd_train.faq_id):
                self.trainid.append(i)
            else:
                self.valtestid.append(i)
        binarytag_table = np.zeros( (len(data_records), len(tag_w2i)))
        binarytag_table[ row, col ] = 1
        self.nq_bi = len(tag_w2i) #120
        self.nq_total = self.nq_bi
        print('there are in total {} binary tags'.format(len(tag_w2i)))
        
        if self.args['using_categorical']:
            self.categorical = []
            self.catq = list(data_records[0]['catq'].keys())
            for cq in self.catq:
                cat = {}
                cat['question'] = cq
                cat['answers'] = []
                cat['idx'] = []
                for i in range(len(data_records)):
                    value = data_records[i]['catq'][cq].lower().strip()
                    tag = cq.lower() + ' ' + value
                    # tag = ' '.join(tag.strip().split(' '))
                    tagid = tag_w2i.setdefault(tag, len(tag_w2i))
                    # if dr['faq_id'] in self.trainid:
                    row.append(i)
                    col.append(tagid)   
                    if tag not in cat['answers']:
                        cat['answers'].append(tag)
                        cat['idx'].append(len(tag_w2i)-1) 
                self.categorical.append(cat)
            print(self.categorical)
            allids =[i for cq in self.categorical for i in cq['idx']]
            print(sorted(allids))

            tagfaq_table = np.zeros((len(data_records), len(tag_w2i)))
            tagfaq_table[ row, col ] = 1.0
            # if self.args['zeroshot']: 
            #     # filtering the goal set.  
            #     for cat in self.categorical:
            #         for i in range(len(data_records)):
            #             if i in valtestid: 
            #                 tagfaq_table[i, cat['idx']] = 1/len(cat['idx'])
            print(tagfaq_table.shape)
            for cat in self.categorical:
                assert np.allclose(tagfaq_table[:, cat['idx']].sum(1), 1.0)
            self.nq_total += (len(self.categorical))
        else:
            tagfaq_table = binarytag_table

        if self.args['zeroshot']: 
            print('resetting the validation test part for expert annotation')
            binary_prior = np.mean(tagfaq_table[self.trainid,:self.nq_bi]) 
            for i in self.valtestid:
                tagfaq_table[i,:self.nq_bi] = binary_prior
                if self.args['using_categorical']:
                    for cat in self.categorical:
                        tagfaq_table[i, cat['idx']] = 1/len(cat['idx'])

        print('there are in total {} tags'.format(len(tag_w2i)))
        tag_i2w = dict((v,k) for k,v in tag_w2i.items())
        features = list(tag_w2i.keys() )

        w2i = {}
        corpus = iqs + candidates + features
        for seq in corpus:
            for w in self.preprocessor.process(seq): 
                if w not in w2i:
                    w2i[w] = len(w2i)
        w2i[self.args['tag_faq_separator'].strip()] = len(w2i)

        self.word_to_index = w2i
        self.tag_i2w = tag_i2w
        self.tag_w2i = tag_w2i
        self.iqs_text = iqs
        self.tgt_ids = tgt_ids
        # sys.exit()
        return tagfaq_table, features, candidates


    def reload_fromprob(self):
        datarecords= self.queryfile.to_dict('records')
        fq_tag_user = np.zeros(self.gold_table.shape)

        if self.args['sampled'] ==2:
            print('\nReading from the second file! \n')
            faq_probs = data_loading.read_turkprob('sampled2/sampled2_')
        else: 
            faq_probs = data_loading.read_turkprob('sampled/sampled_')
            #faq_probs = data_loading.read_turkprob('full/')
        prob_weight = [1, 1, 1, 1, 1]

        if self.args['using_categorical']:
            for i in range( self.gold_table.shape[0]):
                for j in range(self.nq_bi, self.gold_table.shape[1]):
                    fq_tag_user[i, j] = self.gold_table[i,j].copy()
            # print(fq_tag_user[i, self.nq_bi: self.gold_table.shape[1]-1])
            # print(self.gold_table[i, self.nq_bi: self.gold_table.shape[1]-1])
            for cat in self.categorical:
                assert np.allclose(fq_tag_user[:, cat['idx']].sum(1), 1.0)
            fname = datapath+ 'turk_cat_result.json'
            turkprob= json.loads(open(fname, 'r').readline()) 

        for i in range(len( datarecords)):
            dr = datarecords[i]
            tgttext = dr['faqtext'] if 'faqtext' in dr else dr['faq_original']
            tgt_ids = i
            faqid = str(dr['faq_id'])
            labeled = [int(faqid in fp) for fp in faq_probs]
            if not 0 in labeled: 
                for i in range(len(faq_probs)):
                    probdict = faq_probs[i][faqid]
                    for tg in probdict.keys():
                        if tg in self.tag_w2i:
                            fq_tag_user[tgt_ids, self.tag_w2i[tg]] =  probdict[tg]* prob_weight[i]
                            #fq_tag_user[tgt_ids, aa.tag_w2i[tg]] =  int(probdict[tg] >=0.4)
            else:
                print('no data')
                taglist = dr['taglist']
                for tg in taglist:
                    if tg in self.tag_w2i:
                        fq_tag_user[tgt_ids, aa.tag_w2i[tg]] =  1
            if self.args['using_categorical']:
                if faqid in turkprob:
                    turkresult = turkprob[faqid]
                    allcqtag = [r for cat in self.categorical for r in cat['answers']]
                    for tg in allcqtag:
                        if tg in turkresult:
                            fq_tag_user[tgt_ids, self.tag_w2i[tg]] =  turkresult[tg]
                        else:
                            print(tg)
                else:
                    print(faqid)
        
        if self.args['using_categorical']:
            for cat in self.categorical:
                # fq_tag_user[:, cat['idx']] = fq_tag_user[:, cat['idx']]/fq_tag_user[:, cat['idx']].sum(1, keepdims=True)
                assert np.allclose(fq_tag_user[:, cat['idx']].sum(1), 1.0)

        # if self.args['zeroshot']: 
        #     print('resetting the validation test part for expert annotation')
        #     binary_prior = np.mean(fq_tag_user[self.trainid,:self.nq_bi]) 
        #     for i in self.valtestid:
        #         fq_tag_user[i,:self.nq_bi] = binary_prior
        #         if self.args['using_categorical']:
        #             for cat in self.categorical:
        #                 fq_tag_user[i, cat['idx']] = 1/len(cat['idx'])

        return  fq_tag_user


    def get_data(self, data_records):
        #records has fields of 'faq_id', 'faq_original', 'taglist', 'device', 'querylist'
        iqs, tgt_ids =[], []
        for i in range(len(data_records)):
            dr =data_records[i]
            iqs.extend( dr['querylist'])
            tgttext = dr['faqtext'] if 'faqtext' in dr else dr['faq_original']
            #tgttext = dr['faq_original']
            tgtid = self.faqs.index(tgttext)
            tgt_ids += [tgtid]*len(dr['querylist'])
        return np.array(self._preprocess(iqs)), np.array(tgt_ids)


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

    def __iter__(self):
        return self


    def __next__(self):
        position, batch_size = self.position, self.batch_size

        '''
        if position < len(self.iqs):
            # Get queries and positive targets
            queries = self.iqs[position:position + batch_size]
            targets = self.tgt_ids[position:position + batch_size]
        '''
        if position < len(self.iqs_test):
            # Get queries and positive targets
            queries = self.iqs_test[position:position + batch_size]
            targets = self.tgt_test[position:position + batch_size]
            qr_fact = self.faqtag_table[targets]
            
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
        qr_fact = self.faqtag_table[targets]
        return queries, qr_fact, torch.Tensor(targets).long().to(self.device)

    def valdata(self):
        queries = self.iqs_eval  
        targets = self.tgt_eval
        qr_fact = self.faqtag_table[targets]
        return queries, qr_fact, torch.Tensor(targets).long().to(self.device)

    def testdata(self):
        queries = self.iqs_test  
        targets = self.tgt_test
        qr_fact = self.faqtag_table[targets]
        return queries, qr_fact, torch.Tensor(targets).long().to(self.device)



    ## ========================== Encoding, information gain etc ==========================
    def tag_input(self):
        '''
        print('\n\nloading pretrained tag model from {}'.format(self.args['tagckpt']))
        tagmodel  = load_checkpoint(self.args['tagckpt'])[0]
        self.tagmodel = tagmodel.to(self.device)
        '''
        self.tagmodel = self.model
        ft_index = self._preprocess(self.tags)
        ft_embeddings, ft_lengths = self.batch_embedder.embed(ft_index)
        ft_encodings = self.tagmodel.encode(ft_embeddings, ft_lengths).detach()

        # faq_embeddings, faq_lengths = self.batch_embedder.embed(self.faqs_index)
        # faq_encodings = self.tagmodel.encode(faq_embeddings, faq_lengths).detach()
        faq_encodings = self.faqs_mat

        self.faq_encodings = faq_encodings
        self.ft_encodings = ft_encodings

        scores = faq_encodings @ft_encodings.t()

        assert scores.shape ==  self.faqtag_table.shape
        faqtag_belief0 = scores.data
        return faqtag_belief0

    def tag_inference(self):
        if self.args['tag_model'] == 'scalar':
            pafraw = self.faqtag_belief0* self.tagweight + self.tagbias

        if self.args['tag_model'] == 'vector':
            pafraw = self.faqtag_belief0 @ self.tagweight + self.tagbias.unsqueeze(0)

        if self.args['tag_model'] == 'bl':
            pafraw = (self.faq_encodings @ self.tagweight) @ self.ft_encodings.t() + self.tagbias
            pafraw = pafraw.squeeze(0)

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
        belief_t = self.lmda * paf + (1- self.lmda)* torch.Tensor(self.faqtag_table).to(self.device)
        self.faqtag_belief  = belief_t

    ## ========================== Encoding, information gain etc ==========================
    '''
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
            cand_encodings = self.model.encode(cand_embeddings, cand_lengths)
            cand_mat.extend(cand_encodings)
        return torch.stack(cand_mat)
    '''

    def rankbatch(self, queries):
        # self.faqs_mat = utils.encode_candidates(self.faqs_index, self.batch_embedder, self.model, self.batch_size)#self.encode_candidates( self.faqs_index)
        qr_embeddings, qr_lengths = self.batch_embedder.embed(queries)
        query_encodings = self.model.encode(qr_embeddings, qr_lengths)
        score = query_encodings @ self.faqs_mat.t()
        if not self.args['ft_rnn']:
            score = score.detach()
        score = score * self.embedweight
        score = F.softmax(score, -1)  ### Why the softmax actually make a difference 
        return score

    def infogain_batch(self, score, ft_asked=None, debug = False):
        with torch.no_grad():
            p_f_x = score
            # total_entropy = compute_entropy(p_f_x)

            # compute IG of binary questions
            p_a_f = torch.tensor(self.faqtag_belief[:, :self.nq_bi], dtype=torch.float).to(self.device).t()#self.faqtag_belief.t()
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
                print(weight_entropy.shape[0])
                for i in range(weight_entropy.shape[0]):
                    # print(ft_asked[i])
                    mask[i,ft_asked[i]] = 100
                weight_entropy += mask #torch.Tensor(mask).to(self.device)


            best_q = weight_entropy.argmin(dim=1)  # information gain = total_entropy- weighted_entropy

            total_entropy = Categorical(p_f_x).entropy()  #compute_entropy(p_f_x)
            mig = total_entropy- torch.min(weight_entropy, 1)[0]

            ig = torch.stack([mig, total_entropy], 1)
            return best_q #, ig #total_entropy #self.reform_ft(best_f)

def conditional_entropy(p_f_x, p_a_f):
    paf_pfx = p_f_x.unsqueeze(1) * p_a_f 
    p_f_ax = (1e-12 + paf_pfx )/torch.sum(paf_pfx + 1e-12 , 2 , keepdim=True)
    #p_f_ax =  normalize(paf_pfx, p=1, dim=-1)
    new_entropy = Categorical(p_f_ax).entropy() 
    p_a_x = torch.sum(paf_pfx, 2)  
    weighted_entropy = new_entropy * p_a_x
    return weighted_entropy




