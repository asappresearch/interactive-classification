import argparse
import os
import sys
import json
from flask import Flask, render_template, request, jsonify, g, redirect, url_for
import sqlite3

from module.config import  _get_parser
from module.policy import Policy
from module.helpers import  get_params
from module.AskAgent import *
from module.utils import *
from module.cust_simulator import *

app = Flask(__name__)

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(args['db_path'])
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def home():
    sample_faq()
    true_faq = faq_data['true_faq']
    return render_template(
        'home.html',
        faq=true_faq,
        data=faq_data
    )

@app.route('/send/', methods=["POST"])
def get_message():
    message = request.form.get('message')
    if args['save_to_db']:
        db = get_db()
        c = db.cursor()
        c.execute(f"INSERT INTO test (a, b, c) VALUES ('{message}', 'test', 0);")
        db.commit()
    return rollout(message)

@app.route('/faq-details/', methods=["POST"])
def get_faq_details():
    faq_id = request.form.get('faq-id')
    return jsonify(faqs[faq_id])

@app.route('/finish/', methods=['POST'])
def finish():
    global faq_data

    if request.method == 'POST' and args['save_to_db']:
        result = request.form.to_dict()

        keys = ('user_id', 'true_faq', 'question1', 'question2', 'question3', 'question4')
        values = ('NA', faq_data['true_faq'], result['question1'], result['question2'], result['question3'], result['question4'])
        print(faq_data['true_faq'])
        db = get_db()
        c = db.cursor()
        c.execute(f"INSERT INTO questions {str(keys)} VALUES {str(values)};")
        db.commit()
    return render_template('finish.html')

def rollout(message):
    global src, tgt, tgtid, query, queries, faq_data, p_f_a, p_f_x, best_ft, count

    # TODO: we need to do some processing here
    if message in ('yes', 'y', 'yeah'):
        answers = [1]
    elif message in ('no', 'n', 'nope'):
        answers = [0]
    else:
        answers = None
        if query == '':
            query = message
        else:
            query = message + args['tag_faq_separator'] + query
        queries = aa._preprocess([query])
        p_f_x = aa.rankbatch(queries).detach()

    if best_ft and answers: 
        pos_masks, neg_masks, pos_queries = update_with_answer(queries, best_ft, answers, aa, args)
        feedback = pos_masks * neg_masks
        p_f_a = p_f_a * feedback.detach()
    
    p_f_x = normalize(p_f_x * p_f_a , p=1, dim=-1)
    best_ft = aa.infogain_batch(p_f_x)
    best_feature = aa.tag_i2w[best_ft[0].cpu().item()]

    best_faq = aa.faqs[p_f_x.argmax(dim=-1).cpu().item()]
    newscore, indices = torch.sort(p_f_x, dim=-1, descending=True)
    rank = list(indices.data.cpu().numpy()[0]).index(tgtid)

    state = policynet.get_state(p_f_x, count)

    action, log_pact, _ = policynet.select_action(state)
    count += 1

    if action.item() == 0:
        return json.dumps({
            'ft': f'Here is the solution:  {best_faq}', 
            'faq': best_faq, 
            'show_questionnaire': True
        })
    else:
        return json.dumps({
            'ft': f'Does this apply: {best_feature}', 
            'faq': best_faq,
            'show_questionnaire': False
        })
    
def sample_faq(faq=0):
    global src, tgt, tgtid, query, queries, faq_data, p_f_a, p_f_x, best_ft, count
    
    if faq == 0:
        i = np.random.choice(len(aa.iqs_text))
    else:
        i = faq

    src = aa.iqs_text[i]
    tgtid = aa.tgt_ids[i]
    tgt = aa.faqs[tgtid]
    query = ''
    count = 0

    faq_data = {}
    faq_data['true_faq'] = tgt

    tag_onehot = aa.gold_table[tgtid]
    tag_onehot = user.qtag_belief[tgtid]
    faq_data['tag_list'] = [aa.tag_i2w[k] for k in range(len(tag_onehot)) if tag_onehot[k] > 0.5]
    
    p_f_a = torch.ones((1, len(aa.faqs))) 
    best_ft = None


if __name__  =='__main__':
    parser = argparse.ArgumentParser()
    argparser = _get_parser(parser)
    args = argparser.parse_args()
    args = vars(args)
    args['cuda'] = not args.pop('no_cuda')
    args['bidirectional'] = not args.pop('unidirectional')

    if args['tag_pretrain'] == True:
        args['taginfer'] = True
    
    device = torch.device('cuda') if args['cuda'] else torch.device('cpu')

    aa = AskingAgent(args)
    aa.load_state_dict(torch.load('checkpoints/tagft_tagmodelTrue_fttagTrue_ftembTrue_aa.pt'))
    aa = aa.to(device)
    aa.eval()
    
    policynet = Policy(args)
    policynet.model = load_checkpoint('checkpoints/tagft_tagmodelTrue_fttagTrue_ftembTrue_auc05_0.742_e4900.pt')[0]
    policynet = policynet.to(device)
    policynet.model.eval()
    user = PersonaUser(aa, args)

    app.run(port='5000')
