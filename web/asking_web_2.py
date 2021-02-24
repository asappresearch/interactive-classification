from flask import Flask, render_template, request, jsonify

import argparse
import os
import sys

from module.config import  _get_parser
from module.AskAgent import *
from module.helpers import  get_params
from module.utils import *
from module.cust_simulator import *
import json

app = Flask(__name__)

faqs = {
    '17': {'true_faq': 'one', 'tag_list': ['help', 'me', 'im', 'drowning'], 'initial_query': 'hell-oops'}
}

@app.route('/')
def home():
    samplefaq()
    true_faq = faq_data['true_faq']
    options = [
        {'label': 'one', 'text': 'one', 'value': 21},
        {'label': 'two', 'text': 'tween', 'value': 17},
    ]    
    return render_template(
        "home.html",
        faq=true_faq,
        options = options,
        query = query+'?',
        data = faq_data
    )

@app.route('/send/', methods=["POST"])
def get_message():
    message = request.form.get('message')
    # save message, turker_id, and turn to DB
    return rollout(message)

@app.route('/faq-details/', methods=["POST"])
def get_faq_details():
    faq_id = request.form.get('faq-id')
    return jsonify(faqs[faq_id])


def rollout(message):
    global aa, src, tgt, tgtid, query, queries, faq_data, p_f_a, p_f_x, best_ft, user

    if message == 'yes':
        answers  = [1]
    elif message =='no':
        answers = [0]
    else:
        answers = None
        if query == '':
            query = message
        else:
            query = message + args['tag_faq_separator'] + query
        #queries1 = [[aa.word_to_index[w] for w in aa.preprocessor.process(query)]]
        queries = aa._preprocess([query])
        #print(queries1)
        print(queries)
        p_f_x = aa.rankbatch(queries).detach()

    #if message!= '':
    if best_ft and answers: 
        pos_masks, neg_masks, pos_queries = update_with_answer( queries,  best_ft, answers , aa, args)
        feedback = pos_masks * neg_masks
        p_f_a = p_f_a* feedback.detach()


    
    p_f_x  = normalize(p_f_x * p_f_a , p=1, dim=-1)
    best_ft = aa.infogain_batch( p_f_x)
    best_feature = aa.tag_i2w[best_ft[0].cpu().item()]

    best_faq = aa.faqs[p_f_x.argmax(dim=-1).cpu().item()]
    newscore, indices = torch.sort(p_f_x, dim=-1,  descending=True)
    rank = list(indices.data.cpu().numpy()[0]).index(tgtid)
    print(newscore[0][0])
    #if newscore[0][0]>0.99:
    #if rank == 0 and np.random.binomial(1, 0.8):
    assert  newscore[0][0]== max( newscore[0])
    #if np.random.binomial(1, 0.8):
    #if rank == 0 and np.random.binomial(1, 0.6):
    #    return json.dumps({'ft':'Here is the solution:  '+best_faq, 'faq':best_faq} ) 
    if newscore[0][0]>0.9:
        return json.dumps({'ft':'Here is the solution:  '+best_faq, 'faq':best_faq} )
        #TODO: redirect to new page with questions 
    else:
        #return 'RK:{} == FEATURE: {} == FAQ: {}'.format(rank,  best_feature, best_faq)
        return json.dumps({'ft': 'Does this apply: ' +best_feature, 'faq':best_faq} )        

    
def samplefaq(faq=0):
        global aa, src, tgt, tgtid, query, queries, faq_data, p_f_a, p_f_x, best_ft, user
        vlist = [2602, 1427,  2110, 892, 2004, 632, 1980,  1090,  2540, 1595, 1045, 2862,  3608,3351,2155,434]
        
        if faq == 0:
            i = np.random.choice(len(aa.iqs_text))
            #i = np.random.choice(vlist)
        else:
            i = faq
        print(i)
        src = aa.iqs_text[i]
        tgtid = aa.tgt_ids[i]
        tgt = aa.faqs[tgtid]
        query = src 
        query = ''

        faq_data ={}
        faq_data['true_faq'] = tgt
        #faq_data['tag_list'] = [x.strip() for x in src.split(',') if x.strip()!='']

        tag_onehot = aa.gold_table[tgtid]
        tag_onehot = user.qtag_belief [tgtid]
        faq_data['tag_list'] = [aa.tag_i2w[k] for k in range(len(tag_onehot)) if tag_onehot[k]>0.5]
        
        p_f_a = torch.ones((1 , len(aa.faqs)))  #.cuda()
        best_ft = None
        '''
        data = aa.dataset
        features = aa.features
        px_name = np.array(aa.candidates)
        candid_mat = aa.candidates_mat
        recompute = True
        '''

    #new_results=px_name
    #best_feature_index=0
    #best_feature = features[best_feature_index]


if __name__  =='__main__':

    global aa, user, args

    parser = argparse.ArgumentParser()
    argparser = _get_parser(parser)
    args = argparser.parse_args()
    args = vars(args)
    args['cuda'] = not args.pop('no_cuda')
    args['bidirectional'] = not args.pop('unidirectional')

    for k,v in args.items():
        if v == 'False':
            args[k]= False 
        if v == 'True':
            args[k]= True  
    if args['tag_pretrain'] == True:
        args['taginfer'] = True
    print(args)

    aa = AskingAgent(args)
    user = PersonaUser(aa, args)

    app.run(port='5000')
    #app.run()
