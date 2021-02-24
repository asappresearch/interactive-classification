import argparse

import torch
import torch.nn as nn
from typing import Dict, Iterator, Tuple, Union
from tqdm import tqdm, trange
import os
import numpy as np

from collections import defaultdict


from asapp.ml_common.embedders import FastTextEmbedder
# from asapp.ml_common.preprocessors import SplitPreprocessor
from asapp.ml_common.embedders import IndexBatchEmbedder, WordBatchEmbedder 
# from asapp.ml_common.interfaces import  Embedder, Preprocessor


from torch.nn.functional import normalize



def infogain_rollout( batch, aa, user, args, mode):
    device = torch.device('cuda') if args['cuda'] else torch.device('cpu')

    queries, _, targets = batch
    batch_s = len(batch[0])

    cnt =1
    max_step = args['max_step']

    batch_rank = []
    p_fx_batch = []
    score_batch = []
    ft_asked = []
    #p_f_a = np.ones((batch_s , len(aa.clname))) # p_a_f
    p_f_a = torch.ones((batch_s , len(aa.clname))).to(device)

    while cnt <=  max_step:

        p_f_x = aa.rankbatch(queries)
        if not args['ft_emb']:
            p_f_x = p_f_x.detach()
        p_f_x  = normalize(p_f_x * p_f_a , p=1, dim=-1)

        newscore, indices = torch.sort(p_f_x, dim=-1,  descending=True)
        ranks = (indices==targets.unsqueeze(1)).nonzero()[:,1].data.cpu().numpy()
        
        # ================== do the ASK and update chat/belief state =================
        if args['strategy']  =='infogain':
            if cnt !=1 and args['no_rpt_ft']:
                #ft_id = np.array(ft_asked).transpose()
                #print(ft_id.shape)
                best_ft = aa.infogain_batch( p_f_x, ft_asked)
            else:
                best_ft = aa.infogain_batch( p_f_x)

        elif args['strategy'] == 'random':
            best_ft = torch.randint(0,len(aa.nq_bi)+len(aa.categorical), (batch_s,1), dtype=torch.long).squeeze(1)
        elif args['strategy'] =='tag_baseline':
            masked_uniform_prob = normalize(torch.ones_like( p_f_x)* p_f_a, p=1, dim=-1)
            best_ft = aa.infogain_batch(masked_uniform_prob )
        else:
            print('asking module strategy not available')

        if args['using_categorical']:
            answers, answered_ft = user.answer_label_cat(best_ft, batch, aa, mode) 
        else:
            answers, answered_ft = user.answer_label(best_ft, batch, mode)
            
        pos_masks, neg_masks, pos_queries = update_with_answer( queries,  best_ft, answers , aa, args)
        if cnt==1:
            ft_asked = answered_ft
        else:
            newasked = []
            for i in range(len(best_ft)):
                newasked.append(ft_asked[i] + answered_ft[i])
            ft_asked = newasked[:]

        if args['pfx_factorization'] == 'full':
            feedback = pos_masks * neg_masks
        elif args['pfx_factorization'] == 'half':
            feedback = neg_masks
            queries = pos_queries
        elif args['pfx_factorization'] == 'bad':
            feedback = pos_masks * neg_masks
            queries = pos_queries
        else:
            print('pfx factorization not implemented !')
            sys.exit()
        if not args['ft_tag']:
            feedback = feedback.detach()
        p_f_a = p_f_a* feedback
        cnt+=1

        batch_rank.append(ranks)
        p_fx_batch.append(p_f_x)
        score_batch.append(newscore.data.cpu().numpy()) #5*500*517

    return batch_rank, p_fx_batch, np.array(score_batch)



def update_with_answer(queries, best_ft, answers, aa, args):
    device = torch.device('cuda') if args['cuda'] else torch.device('cpu')
    newqp = []
    pos_masks = []
    neg_masks = []
    for i in range(len(best_ft)):
        a = answers[i]
        if a==1:
            q =aa.tag_i2w[best_ft[i].cpu().item()]
            newqp.append( [aa.word_to_index.get(w, len(aa.word_to_index)) for w in aa.preprocessor.process(q)])
        else: 
            newqp.append([])
        if a==1: 
            mk = aa.faqtag_belief[:,best_ft[i]]
            mk += args['belief_reduce']*mk
            pos_masks.append(mk)
            neg_masks.append(torch.Tensor( [1]* len(aa.clname)).to(device))
            #mk = torch.Tensor( [1]* len(aa.clname)).cuda()
        elif a==0:
            mk = 1 - aa.faqtag_belief[:,best_ft[i]]
            mk += args['belief_reduce']*mk
            pos_masks.append(torch.Tensor( [1]* len(aa.clname)).to(device))
            neg_masks.append(mk)
        elif a==-1:
            pos_masks.append(torch.Tensor( [1]* len(aa.clname)).to(device))
            neg_masks.append(torch.Tensor( [1]* len(aa.clname)).to(device))
        else: 
            print(f'multi choice answer: {a}')
            mk = aa.faqtag_belief[:, a]
            pos_masks.append(mk)
            neg_masks.append(torch.Tensor( [1]* len(aa.clname)).to(device))

    #print(pos_masks)
    pos_masks = torch.stack(pos_masks , 0)
    neg_masks = torch.stack(neg_masks , 0)

    if np.array(newqp).size !=0:
        #queries = [newqp[i] + [aa.word_to_index[args['tag_faq_separator'].strip()]] + queries[i] for i in range(len(best_ft))]
        queries = [newqp[i] + queries[i] for i in range(len(best_ft))]
    return pos_masks, neg_masks, list(queries)





def load_checkpoint(checkpoint_path: str):
    """
    Loads a model checkpoint with the model on CPU and in eval mode.

    :param checkpoint_path: Path to model checkpoint to load.
    :return: Returns a tuple with the model, optimizer, epoch number, iteration number, and auc@0.05 of the loaded model.
    """
    # Load state
    state = torch.load(checkpoint_path, map_location=lambda storage, location: storage)
    #state['model'].eval()

    return state['model_state'], state['optimizer'], state['epoch'], state['iter_count'], state['auc05'], state['config']


def load_model(checkpoint_path: str):
    """
    Loads a model with the model on CPU and in eval mode.

    :param checkpoint_path: Path to model checkpoint to load.
    :return: Returns the loaded model on CPU in eval mode.
    """
    return load_checkpoint(checkpoint_path)[0]



def save_nnmodel(model: nn.Module,
                metrics: float,
                epoch: int,
                prev_save_path: str = None) -> str:
    """
    Saves a model checkpoint, including all model parameters and configurations and the current state of training.

    :param model: An nn.Module to save.
    :param epoch: The current epoch count.
    :param metric: The validation auc@0.05 scored by this model.
    :param prev_save_path: The path to the previous model checkpoint in order to delete it.
    :return: The path the the model checkpoint just saved.
    """
    # Generate save path
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    save_path = os.path.join(config['checkpoint_dir'], '{}_metrics_{:.3f}_e{}.pt'.format(config['flavor'], metrics, epoch))
    # Save checkpoint and remove old checkpoint
    print('Saving checkpoint to {}'.format(save_path))
    torch.save(model.state_dict(), save_path)
    if prev_save_path is not None:
        os.remove(prev_save_path)
    return save_path


def save_checkpoint(model: nn.Module,
                    optimizer,
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
        #'model': model,
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



def att2label():
    label2att={}
    att2label=[]
    # labellist=[]
    with open('../CUB_data/attributes/attributes.txt') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            line = lines[i]
            line = line.strip().split()[1].split('::')
            label = line[0]
            if label not in label2att:
                label2att[label]=[i]
                # labellist.append(label)
            else:
                label2att[label].append(i)
            att2label.append(len(label2att)-1)
    return label2att, att2label


def get_att_dict():
    attnames = open('../CUB_data/attributes/mod_att.txt').readlines()
    att2id = {}
    for att in attnames:
        #tagkey = att.split(' ')[0]
        tag = ' '.join(att.split(' ')[1:]).strip()
        att2id[tag] = len(att2id)
    id2att ={v:k for k,v in att2id.items()}
    return att2id, id2att
            


def parse_cat_binary():
    categorical = []
    binary_idx = []

    # labellist=[]
    from collections import defaultdict
    allqr = {}
    attnames = open('../CUB_data/attributes/attributes.txt').readlines()
    for i,line in enumerate(attnames):
            line = line.strip().split()[1].split('::')
            q , r =line[0], line[1]
            if q not in allqr:
                allqr[q] = {'r':[], 'ridx':[]}
            allqr[q]['r'].append(r)
            allqr[q]['ridx'].append(i)

    for q,rs in allqr.items():
        if len(rs['r']) >5:
            binary_idx += rs['ridx']
        else:
            cat ={}
            cat['question'] = q
            cat['answers'] = rs['r']
            cat['idx'] = rs['ridx']
            categorical.append(cat)

    tagnew = binary_idx + [idd  for cq in categorical for idd in cq['idx'] ]
    tag_new2old={k:v for k,v in enumerate(tagnew)}
    tag_old2new={v:k for k,v in enumerate(tagnew)}
    binary_idx = [tag_old2new[idx] for idx in binary_idx]
    for i in range(len(categorical)):
        categorical[i]['idx']= [tag_old2new[idx] for idx in categorical[i]['idx']]
    return binary_idx, categorical, tag_new2old, tag_old2new


def encode_candidates(data, batch_embedder, model, batch_size):
    cand_mat = []
    batch_size = batch_size
    num_batch = len(data) // batch_size

    for batch_idx in trange(num_batch + 1):
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size
        batch = data[start_idx:end_idx]

        if len(batch) == 0:
            break

        cand_embeddings, cand_lengths = batch_embedder.embed(batch)
        cand_encodings = model.encode_class(cand_embeddings, cand_lengths)
        cand_mat.extend(cand_encodings)
    return torch.stack(cand_mat)

