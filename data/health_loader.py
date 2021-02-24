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
import csv
from typing import Dict, Iterator, Tuple, Union, List,  Any
from collections import defaultdict

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# from data.dataset import FaqDataset
from data.dataiterator import DataIterator


datadir = '../interactive/health_data/'


def load_data(config, preprocessor):
    data, label2text = load_healthdata()
    disease_pool = list(label2text.values())
    disease_symptom = read_health_schema()
    query_train, query_val, query_test = split_data(data)

    train_queries, train_targets= processtrain( query_train, label2text , disease_symptom)
    val_queries, val_targets= processval(query_val, label2text )

    print('There are {} training examples'.format(len(train_queries)))
    print('There are {} validation examples\n\n'.format(len(val_queries)))

    print('examples:')
    print( [list(x) for x in zip(train_queries, train_targets)][:3])
    print( [list(x) for x in zip(val_queries, val_targets)][:3])

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
                                  reuse_negatives=True,
                                  negativepool = disease_pool,
                                  init_word_to_index=word_to_index,
                                  embedder_type=config['embeddertype'])
    print('Validation data includes {:,} examples'.format(val_data.num_examples))
    word_to_index = val_data.get_word_to_index()
    # ================================ test dataset ================================
    test_datalist = defaultdict()
    aucresult = defaultdict()
    recallresult = defaultdict()

    flavor = config['flavor']
    tagrange =[0, 1]
    for initq in [1, 0]:
        for tag_r in tagrange:
            if initq ==0 and tag_r == 0:
                continue
            test_key = '_'+str(initq)+'_'+str(tag_r)
            test_queries, test_targets= processtest(query_test, label2text, disease_symptom, initq, tag_r, repeatN=5)
            print('\nBuilding test dataset, {}'.format(test_key))
            test_data = DataIterator(test_queries,
                                          test_targets,
                                          preprocessor,
                                          config['eval_batch_size'],
                                          config['eval_num_negatives'],
                                          reuse_negatives=True,
                                          negativepool = disease_pool,
                                          init_word_to_index=word_to_index,
                                          embedder_type=config['embeddertype'])
            print('Test data includes {:,} examples'.format(test_data.num_examples))
            word_to_index = test_data.get_word_to_index()
            test_datalist[test_key] = test_data

    print('\nVocabulary size = {:,}'.format(len(word_to_index)))

    return train_data, val_data, test_datalist, word_to_index



def read_health_schema(filepath:str=datadir+'DerivedKnowledgeGraph_final.csv')-> Dict:
    disease_symptom = defaultdict(dict)
    with open(filepath) as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            row = dict(row)
            disease  = row['Diseases'].strip().replace(' ', '_')
            symps = row['Symptoms']
            tmpt = [x.split('(') for x in symps.split('),')  ]  
            tmpt = tmpt[:10]                                                                                                                                                                            
            spt =[s[0].strip() for s in tmpt]           
            prob =[float(s[1].replace(')','')) for s in tmpt]
            prob = [p/sum(prob) for p in prob]   
            disease_symptom[disease]['keywords'] = spt
            disease_symptom[disease]['keywords_prob'] = prob
    return disease_symptom
    # print(sampling_dict)


def processval(data, label2text) :
    src = [x[0] for x in data]
    tgt = [label2text[x[1]] for x in data]
    return np.array(src), np.array(tgt)


def processtrain(data, label2text, disease_symptom, repeatN=5) :
    src = []
    tgt = []
    for x in data:
        query = x[0]
        disease = x[1]
        t = label2text[disease]
        symptoms = disease_symptom[disease]['keywords']
        tagsizes = np.random.choice(np.arange(0, len(symptoms)) , repeatN)
        for tagsize in tagsizes: 
            tagsize = max(1, tagsize)
            s = list(np.random.choice(symptoms, size=tagsize, replace=False))
            s.append(query)
            s = ", ".join(s)
            src.append(s)
            tgt.append(t)
    return np.array(src), np.array(tgt)



def processtest(data, label2text, disease_symptom, include_init, tag_r, repeatN=5) :
    src = []
    tgt = []
    for x in data:
        query = x[0]
        disease = x[1]
        t = label2text[disease]
        symptoms = disease_symptom[disease]['keywords']
        for i in range(repeatN):
            tagsize = max(1, int(len(symptoms) * tag_r))
            s = list(np.random.choice(symptoms, size=tagsize, replace=False))
            if include_init:
                s.append(query)
            s = " , ".join(s)
            src.append(s)
            tgt.append(t)
    return np.array(src), np.array(tgt)


def load_healthdata(filename=datadir+'disease_initialquery.csv'):
    # print('\nReading intial query files')
    query_fields = ['Answer.Utterance1', 'Answer.Utterance2', 'Answer.Utterance3']
    label_field = 'Diseases'
    labeltext_fields = ['Input.fixed_sym', 'Input.fixed_descip']

    initialqueries = []
    labels = []
    label2text = {}

    # filename = datadir+'disease_initialquery.csv'
    csv_reader = csv.DictReader(open(filename, 'r'))
    for row in csv_reader:
        row = dict(row)
        labeltext = ' , '.join([row[field] for field in labeltext_fields])
        label = row[label_field].strip().replace(' ', '_')
        label2text[label] = labeltext.replace('[disease_name]', row[label_field].strip())
        for field in query_fields: 
            initialqueries.append(row[field])
            labels.append(label)
            # labeltexts.append(labeltext)

    data = zip(initialqueries, labels) #, labeltexts)
    data = [list(x) for x in data]
    print(f'There are in total {len(data)} initial queries')
    return data, label2text 



def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = string.encode('ascii', 'ignore').decode('ascii')
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()





def get_intent_tag_table(targets, filepath):
    ### read corpus and vocab
    disease_symptom = read_health_schema(filepath)
    intent_binaryq = disease_symptom

    tag_w2i = {}
    row, col = [], []
    # Read the binary question answers 
    for i, intent in enumerate(targets):
        kwlist = intent_binaryq[intent]['keywords']
        for kw in kwlist:
            kw = kw.lower().strip()
            row.append(i)
            col.append(tag_w2i.setdefault(kw, len(tag_w2i)))

    binq = list(tag_w2i.keys())

    # categorical = []
    # for cq in catq:
    #     cat = {}
    #     cat['question'] = cq
    #     cat['answers'] = []
    #     cat['idx'] = []
    #     for i, intent in enumerate(targets):
    #         value = intent_catq[intent][cq].lower().strip()
    #         tag = cq.lower().strip() + ' ' + value
    #         row.append(i)
    #         col.append(tag_w2i.setdefault(tag, len(tag_w2i)))   
    #         if tag not in cat['answers']:
    #             cat['answers'].append(tag)
    #             cat['idx'].append(len(tag_w2i)-1) 
    #     categorical.append(cat)

    intenttag_table = np.zeros((len(targets), len(tag_w2i)))
    intenttag_table[ row, col ] = 1.0

    # for cat in categorical:
    #     print(cat['question'])
    #     assert np.allclose(intenttag_table[:, cat['idx']].sum(1), 1.0)
    # print(f'whole table shape {intenttag_table.shape}')
    # print(intenttag_table.sum(1))
    
    #### reaload prob:
    prob_dict = disease_symptom
    for tgt in prob_dict : 
        for kw, prob in zip(prob_dict[tgt]['keywords'], prob_dict[tgt]['keywords_prob']):
            if tgt in targets and kw in tag_w2i:
                intenttag_table[targets[tgt], tag_w2i[kw]] = prob
            else: 
                print(f'keywords not in table: tgt:{tgt}; kw: {kw}')
    categorical, catq = [],[]
    return intenttag_table, tag_w2i, binq, categorical, catq


# def reload_prob(prob_dict, intenttag_table, targets, tag_w2i):
#     for tgt in prob_dict: 
#         for kw, prob in zip(prob_dict[tgt]['keywords'], prob_dict[tgt]['keywords_prob']):
#             if tgt in targets and kw in tag_w2i:
#                 intenttag_table[targets.index(tgt), tag_w2i[kw]] = prob
#             else: 
#                 print(f'keywords not in table: tgt:{tgt}; kw: {kw}')
#     return intenttag_table




def split_data(data: List[Any],
               sizes: Tuple[float, float, float] = (0.8, 0.1, 0.1),
               seed: int = 0) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Randomly splits data into train, val, and test sets according to the provided sizes.
    :param data: The data to split into train, val, and test.
    :param sizes: The sizes of the train, val, and test sets (as a proportion of total size).
    :param seed: Random seed.
    :return: Train, val, and test sets.
    """
    # Checks
    assert len(sizes) == 3
    assert all(0 <= size <= 1 for size in sizes)
    assert sum(sizes) == 1

    # Shuffle
    random.seed(seed)
    random.shuffle(data)

    # Determine split sizes
    train_size = int(sizes[0] * len(data))
    train_val_size = int((sizes[0] + sizes[1]) * len(data))

    # Split
    train = data[:train_size]
    val = data[train_size:train_val_size]
    test = data[train_val_size:]

    return train, val, test
