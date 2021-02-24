import argparse

import torch
from typing import Dict, Iterator, Tuple, Union

from tqdm import tqdm, trange
import os
import numpy as np

from collections import defaultdict

from utils.config import _get_parser


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

    #p_f_a = np.ones((batch_s , len(aa.faqs))) # p_a_f
    p_f_a = torch.ones((batch_s , len(aa.faqs))).to(device)

    while cnt <=  max_step:

        p_f_x = aa.rankbatch(queries)
        if not args['ft_emb']:
            p_f_x = p_f_x.detach()
        p_f_x  = normalize(p_f_x * p_f_a , p=1, dim=-1)

        newscore, indices = torch.sort(p_f_x, dim=-1,  descending=True)
        ranks = (indices==targets.unsqueeze(1)).nonzero()[:,1].data.cpu().numpy()
        
        # ================== do the ASK and update chat/belief state =================
        if args['strategy']  =='infogain':
            best_ft = aa.infogain_batch( p_f_x)
        elif args['strategy'] == 'random':
            best_ft = torch.randint(0,aa.nq_total, (batch_s,1), dtype=torch.long).squeeze(1)
        elif args['strategy'] =='tag_baseline':
            masked_uniform_prob = normalize(torch.ones_like( p_f_x)* p_f_a, p=1, dim=-1)
            best_ft = aa.infogain_batch(masked_uniform_prob )
        else:
            print('asking module strategy not available')

        if args['using_categorical']:
            answers = user.answer_cat(best_ft, batch, aa, mode)
        else:
            answers = user.answer(best_ft, batch, mode)        
        
        pos_masks, neg_masks, pos_queries = update_with_answer( queries,  best_ft, answers , aa, args)

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
        if a:
            q =aa.tag_i2w[best_ft[i].cpu().item()]
            newqp.append( [aa.word_to_index.get(w, len(aa.word_to_index)) for w in aa.preprocessor.process(q)])
        else: 
            newqp.append([])
        if 0< a <= 1.0:
            mk = aa.faqtag_belief[:,best_ft[i]]
            # mk += args['belief_reduce']*mk
            pos_masks.append(mk)
            neg_masks.append(torch.Tensor( [1]* len(aa.faqs)).to(device))
            #mk = torch.Tensor( [1]* len(aa.faqs)).cuda()
        elif a==0:
            mk = 1 - aa.faqtag_belief[:,best_ft[i]]
            # mk += args['belief_reduce']*mk
            pos_masks.append(torch.Tensor( [1]* len(aa.faqs)).to(device))
            neg_masks.append(mk)
        else: 
            print(f'multi choice answer: {a}')
            mk = aa.faqtag_belief[:, a]
            pos_masks.append(mk)
            neg_masks.append(torch.Tensor( [1]* len(aa.faqs)).to(device))

    pos_masks = torch.stack(pos_masks , 0)
    neg_masks = torch.stack(neg_masks , 0)

    if np.array(newqp).size !=0:
        queries = [newqp[i] + [aa.word_to_index[args['tag_faq_separator'].strip()]] + queries[i] for i in range(len(best_ft))]
    return pos_masks, neg_masks, list(queries)


def reprocess_withmask(action_batch, rank_batch, logp_batch, device, args):
    batch_size = len(action_batch[0])
    # ============== Reshape rewards and logp with the mask ============== 
    action_mask=[]
    reward_batch =[]
    
    active_trace = np.ones( batch_size )  
    for i in range(args['max_step']):
        action_mask.append(np.copy(active_trace))
        action = action_batch[i]
        correct = rank_batch[i]==0
        reward_faq = args['reward_p']*correct  + args['reward_n']*(1-correct)
        reward_t = reward_faq*(1-action) + args['reward_ask']*action
        reward_t = reward_t * active_trace
        reward_batch.append(reward_t)
        active_trace *= action
        
    r_mask = np.array(action_mask).transpose()
    logp_bs = torch.stack(logp_batch, 1)*torch.Tensor(r_mask).to(device)

    # ============== Recalcuate the reward using gamma, take the rollout reward ============== 
    gamma = args['gamma']
    R = np.zeros(batch_size)
    for i in range(args['max_step'])[::-1]:
        r_col = reward_batch[i]
        R =r_col + R*gamma

    rewards = torch.FloatTensor(R).to(device)  
    
    # ============== Calculate the states ============== 
    expectedR = torch.sum(rewards* torch.exp( torch.sum(logp_bs, 1))).item()/batch_size 
    reward_batch = np.array(reward_batch)
    assert (reward_batch == args['reward_p']).sum() + (reward_batch == args['reward_n']).sum() == batch_size 
    suc_rate =  (reward_batch  == args['reward_p']).sum()/batch_size 
    ave_turns = np.sum(r_mask ).item()/batch_size 
    scalars = [('/expectedR', expectedR), ('/ave_turns', ave_turns), ('/suc_rate', suc_rate)]

    return rewards,logp_bs, scalars


def recall_fromranks( ranks, ks = [1, 2, 3, 5]):
    recalls  = defaultdict(list)
    for i in range( len(ranks[0]) ):
        rank_i = np.array(ranks)[:, i]
        for k in ks:
            a = rank_i < k
            recalls[k].append(sum(a)/len(a))
    print('\nrecall@k:')
    for k, v in recalls.items():
        print(','.join([str(k)] + [str(x) for x in v]))
        #fout.write(','.join([str(k)] + [str(x) for x in v]))
    return recalls



def suc_withthreshold( ranks, allscore,  ths = [ 0.5, 0.6, 0.7, 0.8, 0.9]):
    topscores = allscore[:,:, 0]

    for th in ths:
        suc = 0
        R3=0
        turns = 0
        scores = topscores.copy( )
        rk = ranks.copy( )
        for i in range(0, ranks.shape[1] -1):
            
            if not scores.shape[0]:
                break
            idx = np.where(scores[:, i] > th)
            suc += np.sum( rk[idx,i]==0  )
            turns += (i+1)*len(idx[0])
            #print(len(idx[0]))

            scores = np.delete(scores, idx, 0)
            #print(scores.shape)
            rk = np.delete(rk, idx, 0)
        suc += np.sum( rk[:,-1]==0  )
        turns += ranks.shape[1]*len(rk)
        print('threshold :{}, turns:{}, sucrate: {}'.format(th,turns/ranks.shape[0], suc/ranks.shape[0]))



def parsefaq_dec( faqs, initialq, tag_l, tag_h,  sampleN=5):
    newfaqs =[]
    src =[]
    tgt=[]
    for i, faq in enumerate(faqs): 
        faq_reform ={}
        tl1_list = [x.strip().strip("'") for x in str(faq['topic_level1']).strip('[]').split(',')]
        tl2_list = [x.strip().strip("'") for x in str(faq['topic_level2']).strip('[]').split(',')]
        actionlist = faq["action"].split(',')
        rt = faq["related topic (noun)"].split(',')

        faq_reform['faq'] = faq['faq_original']
        tags = [x.strip().lower() for x in tl1_list + tl2_list  + actionlist + rt 
                                   +  [faq['type']] +[faq['device']] if x!='']
        tags = list(set(tags))
        faq_reform['binarytag'] = tags
        #faq_reform['catgoricaltag'] = [faq['device']] #+ tl1_list 
        faq_reform['queries'] = [faq['question_0']] +[faq['question_1']]
        faq_reform['index'] = i
        newfaqs.append(faq_reform) 

        for i in range(sampleN):
            t =  faq['faq_original']
            s =[]
            if initialq: 
                s.append(random.choice(faq_reform['queries']))
            tagsize = int(len(tags) *np.random.uniform(tag_l, tag_h))
            s = ", ".join(s + list(np.random.choice(tags, size=tagsize, replace=False)))
            src.append(s)
            tgt.append(t)

    return np.array(src), np.array(tgt), newfaqs


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
        if not cand_encodings.requires_grad:
            cand_mat.extend(cand_encodings.cpu().numpy())
        else:
            cand_mat.extend(cand_encodings)
        del cand_encodings
        del cand_embeddings
    device = torch.device('cuda')  if torch.cuda.is_available() else torch.device('cpu')
    if  isinstance( cand_mat[0], np.ndarray ):
        cand_mat = torch.Tensor(np.array(cand_mat)).to(device)
    else:
        cand_mat = torch.stack(cand_mat)

    return cand_mat


def rerank(query, candidates_mat, batch_embedder, model):
    qr_embeddings, qr_lengths = batch_embedder.embed([query])
    query_encodings = model.encode_context(qr_embeddings, qr_lengths)
    query_encodings = query_encodings.repeat(len(candidates_mat), 1)
    # print(type(query_encodings))
    # print(type(candidates_mat))
    scores = model.score(query_encodings, candidates_mat, scoring_method='dot product')
    scores = scores.data.cpu().numpy()

    ranks = scores.argsort()[::-1]
    
    new_scores = [scores[idx] for idx in ranks]

    return ranks, new_scores


def evaluate_recall(y, y_test, k=1):
    num_examples = float(len(y))
    num_correct = 0
    for predictions, label in zip(y, y_test):
        if label in predictions[:k]:
            num_correct += 1

    return num_correct / num_examples



def recall(val_data, batch_embedder, model, batch_size, ks=[1, 2, 3, 5, 10, 100], verbose=False):
    """Computes recall at k for actual agent reply when added to whitelist"""

    y_true = []
    y_rank = []
    with torch.no_grad():
        candidates_mat = encode_candidates(val_data.negativepool, batch_embedder, model, batch_size)
        # candidates_mat = candidates_mat.to('cuda')

    for i, (query, target) in enumerate(zip(val_data.queries, val_data.targets)):
        ranks, new_scores = rerank(query, candidates_mat, batch_embedder, model)

        
        if verbose:
            new_results = [val_data.negative_text[idx] for idx in ranks]

            print("\n\n==============")
            print('The query is:\n{}\n'.format(val_data.queries_text[i]))
            print('The correct answer is:\n{}\n'.format(val_data.targets_text[i]))

            print('The prediction is:')
            for pred, score in zip(new_results[:3], new_scores[:3]):
                print("{}\n{}".format(pred, score))
    

        y_true.append(val_data.negativepool.index(target))
        y_rank.append(ranks)

    results = []
    print('\nRecall evaluation:')
    for k in ks:
        recall = evaluate_recall(y_rank, y_true, k)
        results.append(recall)
        print("Recall @ ({}, {}): {:g}".format(k, len(val_data.negativepool), recall ))

    return results

 

def deprecated_update_X_determined(masks, queries, best_ft, qr_fact, aa,args):
    newqp = []
    this_masks = []

    for i in range(len(best_ft)):
        a = qr_fact[i, best_ft[i]]
        if a:
            q =aa.tag_i2w[best_ft[i].cpu().item()]
            newqp.append( [aa.word_to_index.get(w, len(aa.word_to_index)) for w in aa.preprocessor.process(q)])
        else: 
            newqp.append([])
        if a: 
            mk = aa.faqtag_table[:,best_ft[i]]
            mk += args['belief_reduce']*mk

            #mk = [1]* len(aa.faqs)
        else:
            mk = 1 - aa.faqtag_table[:,best_ft[i]]
            mk += args['belief_reduce']*mk

        this_masks.append(mk)
    this_mask = torch.Tensor(this_masks).cuda() if args['cuda'] else torch.Tensor(this_masks)
    masks *= this_mask

    if np.array(newqp).size !=0:
        #queries = [queries[i]+  newqp[i] for i in range(len(best_ft))]
        queries = [newqp[i] + [aa.word_to_index[args['tag_faq_separator'].strip()]] + queries[i] for i in range(len(best_ft))]
    #if np.array(newqp).size !=0:
    #    queries = np.array(queries) + np.array(newqp)
    return masks, list(queries)
