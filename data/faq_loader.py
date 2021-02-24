import os
from typing import Dict

from tqdm import tqdm


from pathlib import Path
import sys
import pandas as pd
import random
import numpy as np

import ast
import torch
import json
from collections import defaultdict


np.random.seed(345)
random.seed(345)
torch.manual_seed(7)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(7)

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

datapath = '../interactive/faq_data/'

# from data.dataset import FaqDataset
from data.dataiterator import DataIterator

def load_data(config, preprocessor):
    #tagfile = readtagdata()
    queryfile, faq_pool= read_queries(config)
    query_train, query_val, query_test = query_cv(queryfile, fold_n= config['cv_n'])

    train_queries, train_targets= parse_trainfaq( query_train,  no_paradata = config['no_para_data'], notag = config['no_tag_data'], repeatN=5 )
    val_queries, val_targets= parse_trainfaq( query_val,  no_paradata=True, repeatN=5)

    print('There are {} training examples'.format(len(train_queries)))
    print('There are {} validation examples\n\n'.format(len(val_queries)))

    if config['bert']:
        config['embeddertype'] = 'bpe'
    # Build datasets
    # ================================ Train dataset ================================
    print('Building train dataset')
    train_data = DataIterator(train_queries,
                                    train_targets,
                                    preprocessor,
                                    config['batch_size'],
                                    config['num_negatives'],
                                    reuse_negatives=True,
                                    embedder_type=config['embeddertype'])
    print('Training data includes {:,} examples'.format(train_data.num_examples))
    word_to_index = train_data.get_word_to_index()

    print('Building validation dataset')
    val_data = DataIterator(val_queries,
                                  val_targets,
                                  preprocessor,
                                  config['eval_batch_size'],
                                  config['eval_num_negatives'],
                                  reuse_negatives=True, ########## changed because of bert memory issue
                                  negativepool = faq_pool,
                                  init_word_to_index=word_to_index,
                                  embedder_type=config['embeddertype'])
    print('Validation data includes {:,} examples'.format(val_data.num_examples))
    word_to_index = val_data.get_word_to_index()
    # ================================ test dataset ================================
    test_datalist = defaultdict()
    # aucresult = defaultdict()
    # recallresult = defaultdict()

    flavor = config['flavor']
    tagrange =[0,  0.5,  1]
    tagrange =[0, 1]
    for initq in [1, 0]:
        for tag_r in tagrange:
            if initq ==0 and tag_r == 0:
                continue
            if initq ==0 and tag_r == 0.25:
                continue
            test_key = '_'+str(initq)+'_'+str(tag_r)
            test_queries, test_targets= parse_testfaq( query_test, initq, tag_r, repeatN=5)
            print('\nBuilding test dataset, {}'.format(test_key))
            test_data = DataIterator(test_queries,
                                          test_targets,
                                          preprocessor,
                                          config['eval_batch_size'],
                                          config['eval_num_negatives'],
                                          reuse_negatives=True,  ########## changed because of bert memory issue
                                          negativepool = faq_pool,
                                          init_word_to_index=word_to_index,
                                          embedder_type=config['embeddertype'])
            print('Test data includes {:,} examples'.format(test_data.num_examples))
            word_to_index = test_data.get_word_to_index()
            test_datalist[test_key] = test_data

    print('\nVocabulary size = {:,}'.format(len(word_to_index)))

    return train_data, val_data, test_datalist, word_to_index




def read_turkprob(path):
    fpath = datapath +'paf_anno_result/'+path
    faq_probs = []
    for i in range(5):
        fname = fpath+ 'faq_probs_{}to{}.json'.format(10*i , 10*i+10)
        prob= json.loads(open(fname, 'r').readline()) 
        faq_probs.append(prob)
    return faq_probs



def readtagdata(config):
    # ====== paraphrase data======
    '''
    tag = gc.open("sub_tagfile_518_1129").sheet1
    #faq_labelled_0819_tags_all  ; faq_labelled_v1015
    tags_df = pd.DataFrame.from_records( tag.get_all_records() ).replace(np.nan, '', regex=True)
    '''
    #print(tags_df.columns)
    #tags_df['faq_original'] = tags_df[['title', 'question']].apply(lambda x: ' , '.join(x), axis=1)
    def mergetag(faq):
        listfields= ['topic_level1', 'topic_level2']
        fields = ['action', 'related topic (noun)',  'type', 'type2' ] #, 'device']
        taglist=[]
        if listfields:
            for fd in listfields:
                taglist += ast.literal_eval(faq[fd])
        for fd in fields:
            taglist += faq[fd].split(',')
        taglist =[x.strip() for x in taglist if x !='' ]
        taglist = list(set(taglist))
        return taglist

    categorical=config['using_categorical']
    if categorical:
        tags_df = pd.read_csv(datapath+'sub_tagfile_518_1129-redo_august.csv').replace(np.nan, '', regex=True)
        tags_df = tags_df[tags_df.remove!=1]
        tags_df['taglist'] = tags_df.apply(mergetag, axis=1)
        categiricalq = ['general service type', 'device service', 'sprint service', 'phone service']    
        def getcat(faq):
            intent_catgoricalq = {cq: faq[cq] for cq in categiricalq}
            return intent_catgoricalq
        tags_df['catq'] = tags_df.apply(getcat, axis=1)
        tagfile = tags_df[['faq_id', 'faqtext', 'faq_original', 'taglist', 'device', 'catq']]
        return tagfile

    if config['inputf'] == 'old':
        print('reading old tag file')
        tags_df = pd.read_csv(datapath+'sub_tagfile_518_1129 - sub_tagfile.csv').replace(np.nan, '', regex=True)
    if config['inputf'] == 'new':
        print('reading new new file')
        tags_df = pd.read_csv(datapath+'sub_tagfile_518_1129-redo_august.csv').replace(np.nan, '', regex=True)    
    tags_df['taglist'] = tags_df.apply(mergetag, axis=1)
    tagfile = tags_df[['faq_id', 'faqtext', 'faq_original', 'taglist', 'device']]
    print(f'{len(tagfile)} rows in tagfile')
    print('parsing the tagfile and get fields: {}\n'.format(tagfile.columns))
    return tagfile




def read_queries(config):
    print('\nReading intial query files')
    '''  
    turkfile = 'initialquery_all_1129'
    init = pd.DataFrame.from_records( gc.open(turkfile).sheet1.get_all_records())
    '''
    init = pd.read_csv(datapath+'initialquery_all_1129 - faqs_mturk_20180925_initial_query_batch535_assign3_recovered.csv')

    print('Theare are {} initial queries'.format(len(init)))
    print( '<_device_> token fixed in the user queries' )
    init['iq'] = init.apply(lambda x: x['question'].strip().replace('<_device_>' , 'phone'),axis=1)
    init= init[['faq_id', 'iq']]
    # put queries from different turkers into a list
    initq= init.groupby(['faq_id']).apply(lambda x: x['iq'].tolist()).reset_index()
    initq.rename(columns = {0:'querylist'}, inplace = True)
    
    '''
    # read the original faq list map
    tq_dedupe =  pd.DataFrame.from_records( gc.open("reformed_faq_nodevice").sheet1.get_all_records() )
    tq_dedupe = tq_dedupe[['faq_id', 'faqtext']]
    # Merge files and expand the faqid
    queryfile = initq.join(tq_dedupe.set_index(['faq_id']), on='faq_id',how='inner')
    '''

    tagfile = readtagdata(config)
    queryfile = initq.join(tagfile.set_index('faq_id'), on='faq_id', how='inner')

    print('Get total {} UNIQUE FAQs with iniqial queries'.format(len(queryfile)))
    print('After parsing, file with fields: {}\n'.format(queryfile.columns))

    faq_pool = queryfile.faq_original.tolist()
    return queryfile, faq_pool


def adding_paradata(initq):
    print('\nReading paraphrase dataset')
    '''
    p0 = gc.open("spear-mturk.dev.pos1129").sheet1
    paras_df= pd.DataFrame.from_records( p0.get_all_records())[['faq_id','question']]
    '''
    paras_df = pd.read_csv(datapath+'spear-mturk.dev.pos1129 - spear-mturk.train.pos.csv').replace(np.nan, '', regex=True)[['faq_id','question']]

   
    # put queries from different turkers into a list
    paras= paras_df.groupby(['faq_id']).apply(lambda x: x['question'].tolist()).reset_index()
    paras.rename(columns = {0:'paralist'}, inplace = True)

    '''
    tag = gc.open("faq_all_original").sheet1
    tags_df = pd.DataFrame.from_records( tag.get_all_records() ).replace(np.nan, '', regex=True).drop([504, 572])
    '''
    tags_df = pd.read_csv(datapath+'faq_all_original - faq_labelled_v1017.csv').replace(np.nan, '', regex=True).drop([504, 572])

    tags_df['faq_original'] = tags_df.apply(lambda x: x['question'] if x['title']=='' else ', '.join(x[['title', 'question']]) , axis=1)
    tagfile =tags_df[['faq_id','device', 'faq_original']]
    
    paras = paras.join(tagfile.set_index('faq_id'), on='faq_id', how='inner')
    paras = paras[['faq_id', 'paralist', 'device', 'faq_original' ]]

    # read the original faq list map
    #tq_dedupe=  pd.DataFrame.from_records( gc.open("faq_templated_device_dedup1129").sheet1.get_all_records() )
    tq_dedupe = pd.read_csv(datapath+'faq_templated_device_dedup1129 - faq_templated_device_dedup_dv.csv')

    tq_dedupe = tq_dedupe[['faq_id', 'faqid_list']]
    # Merge files and expand the faqid
    result =initq.join(tq_dedupe.set_index(['faq_id']), on='faq_id',how='inner')
    #result = result[['faqid_list', 'querylist']]
    result = result.drop('faq_id', 1)
    s = result.apply(lambda x: pd.Series(ast.literal_eval(x['faqid_list'])),axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'faq_id'
    queryfile = result.drop(['faqid_list', 'device', 'faq_original'], axis=1).join(s)  # replace faqid_list with faq_id
    
    query_para = queryfile.join(paras.set_index('faq_id'), on='faq_id', how='inner')

    print('After mathing with paraphrase, get total {} FAQs'.format(len(query_para)))
    print('After parsing, file with fields: {}\n'.format(query_para.columns))
    return query_para


def query_cv(queryfile, fold_n=3, fold_total=5, random_s = 345):
    print('\nsplitting intial query files')
    X = np.arange(len(queryfile))
    kf = KFold(n_splits=fold_total, shuffle=True, random_state=random_s)
    splits = kf.split(X)
    cnt = 0 
    while cnt <= fold_n:
        train_index, test_index =next( splits )
        cnt +=1
    train_index, val_index = train_test_split( train_index, test_size= 1/(fold_total-1), random_state=random_s)
    query_train = queryfile.iloc[train_index]
    query_val = queryfile.iloc[val_index]
    query_test = queryfile.iloc[test_index]
    return query_train, query_val, query_test


#parse_trainfaq( query_train,  no_paradata = False, notag = False, repeatN=5 )
def parse_trainfaq( querydf,  no_paradata = False, notag = False, faq_reform = False, tag_intgt=False, repeatN=5):
    print('\nParsing file using random ratio')
    if not no_paradata:
        querydf = adding_paradata(querydf)
    #querydf = querydf.join(tagfile.set_index('faq_id'), on='faq_id', how='inner')
    #print('Before parsing to src_tgt pairs, file with fields: {}'.format(querydf.columns))
    faqs = np.array(querydf.to_dict('records')) # got fields of 'faq_id', 'faq_original', 'taglist', 'device', 'querylist'

    all_tags = [tag for faq in faqs for tag in faq['taglist']]
    n_alltags = len(set(all_tags))
    print(f'======!!!=======there are {n_alltags} tags in training set')

    src =[]
    tgt=[]
    
    
    for i, faq in enumerate(faqs): 
        faqtext = faq['faqtext'] if faq_reform else faq['faq_original']
        '''
        if 'paralist' in faq: 
            if np.random.binomial(1, config['para_ratio']): 
                faqtext =  random.choice([x for x in faq['paralist'] if x!=initq])
        '''
        t = faqtext if not tag_intgt else ' , '.join([ faqtext ] + faq['taglist'])

        faq['taglist'] = faq['taglist'] + [faq['device']]
        qlist = faq['querylist']+faq['paralist'] if not no_paradata  else faq['querylist']
      
        tagsizes = np.random.choice( np.arange(0, len(faq['taglist'])), repeatN) if not notag else [0]

        for initq in qlist: 
            for tagsize in tagsizes: 
                s = list(np.random.choice(faq['taglist'], size=tagsize, replace=False))
                if np.random.binomial(1,0.8): 
                    s.append(initq)
                if len(s) == 0 :
                        continue
                s = ", ".join(s)
                src.append(s)
                tgt.append(t)
    print('get query pairs : {}'.format(len(src)))
    return np.array(src), np.array(tgt)


def parse_testfaq( querydf, include_init, tag_r , faq_reform = False, tag_intgt=False, repeatN=5):
    #print('Parsing file using chosen ratio')
    #querydf = querydf.join(tagfile.set_index('faq_id'), on='faq_id', how='inner')
    #print('Before parsing to src_tgt pairs, file with fields: {}'.format(querydf.columns))
    faqs = np.array(querydf.to_dict('records')) # got fields of 'faq_id', 'faq_original', 'taglist', 'device', 'querylist'

    src =[]
    tgt=[]
    
    for i, faq in enumerate(faqs): 
        faqtext = faq['faqtext'] if faq_reform else faq['faq_original']
        t = faqtext if not tag_intgt else ' , '.join([ faqtext ] + faq['taglist'])

        faq['taglist'] = faq['taglist'] + [faq['device']]

        if include_init==1 and tag_r ==1:
            for initq in faq['querylist']:
                s = ", ".join(faq['taglist'] + [initq])
                if len(s) == 0 :
                        continue
                src.append(s)
                tgt.append(t)
        
        for initq in faq['querylist']: 
            for k in range(repeatN):
                tagsize = int(len(faq['taglist']) * tag_r)
                if tagsize:
                    s = list(np.random.choice(faq['taglist'], size=tagsize, replace=False))
                else:
                    s= []
                if include_init:
                    s.append(initq)
                if len(s) == 0 :
                        continue
                s = ", ".join(s)
                src.append(s)
                tgt.append(t)
    print('get query pairs : {}'.format(len(src)))
    return np.array(src), np.array(tgt)


def parse_testfaq_device( querydf,  faq_reform = False, tag_intgt=False, repeatN=5):
   
    faqs = np.array(querydf.to_dict('records')) # got fields of 'faq_id', 'faq_original', 'taglist', 'device', 'querylist'
    src =[]
    tgt=[]
    for i, faq in enumerate(faqs): 
        faqtext = faq['faqtext'] if faq_reform else faq['faq_original']
        t = faqtext if not tag_intgt else ' , '.join([ faqtext ] + faq['taglist'])
        for initq in faq['querylist']: 
                s = ", ".join([faq['device']] + [initq])
                if len(s) == 0 :
                        continue
                src.append(s)
                tgt.append(t)
    print('get query pairs : {}'.format(len(src)))
    return np.array(src), np.array(tgt)