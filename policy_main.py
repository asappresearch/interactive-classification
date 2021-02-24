import argparse
from tensorboardX import SummaryWriter
import os
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from sru import SRU


from utils.config import  _get_parser
from utils.utils import *
from utils.helpers import save_checkpoint, get_params, compute_param_norm, compute_grad_norm

from interactive.policy import Policy
from interactive.cust_simulator import PersonaUser, NoisyUser
from interactive.AskAgent import AskingAgent

import sys, os
sys.path.append(os.path.abspath('../'))

def main(args):

    aa = AskingAgent(args)
    
    if args['user_type'] == 'oracle':
        user = NoisyUser(args)
    elif args['user_type'] == 'persona':
        user = PersonaUser(aa, args)
    else:
        print('no user type implemented')

    device = torch.device('cuda') if args['cuda'] else torch.device('cpu')
    print(device)

    writer = SummaryWriter(os.path.join(args['tensorboard_dir'], args['comment']+'_'+args['flavor']))
    writer.add_text('Args', args['comment']+' '+ str(args)+ '\n')
    save_path = args['checkpoint_path']

    #==========loading data =============
    policynet = Policy(args)

    print('policy network model: ')
    print(policynet.model)
    writer.add_text('model' ,  str(policynet.model))
    optimizer = optim.Adam(policynet.model.parameters(), lr=args['lr'] )

    ftparams = []
    if args['ft_tag']:
        ftparams += [aa.tagweight, aa.tagbias, aa.lmda]
    if args['ft_emb']:
        if args['ft_rnn']:
            for m in aa.model.modules():
                if isinstance(m, nn.Dropout):
                    m.p = args['dropout']
                if isinstance(m, SRU):
                    m.dropout = args['dropout']
            ftparams += get_params(aa.model)
        else:
            ftparams += [aa.embedweight]
    if args['ft_emb'] or args['ft_tag']:
        print('Finetuning turned on ')
        nnoptimizer = optim.Adam(ftparams, lr=args['ft_lr'] )
    else:
        nnoptimizer=None


    for episode in range(1, args['episodes']):
        if episode % ( args['test_every']) == 0:
            batch = aa.testdata()
            mode = 'test'
            policynet.model.eval()
            aa.model.eval()

        elif episode % args['eval_every'] == 0:
            batch = aa.valdata()
            mode = 'val'
            policynet.model.eval()
            aa.model.eval()

        else:
            batch = aa.sampletrain(args['batch_size']) 
            mode = 'train'
            policynet.model.train()
            aa.model.train()

        
        batch_s = len(batch[0])
        rank_batch, p_fx_batch, _= infogain_rollout(batch, aa, user, args, mode)
        
        action_batch = []
        logp_batch = []
        for cnt in range( 1, len(p_fx_batch)+1):
            p_f_x = p_fx_batch[cnt-1]
            
            if not args['ft_tag'] and not args['ft_emb']:
                p_f_x = p_f_x.detach()

            if cnt == args['max_step']:
                action = np.zeros(batch_s)
                log_pact = torch.zeros(batch_s).to(device)
            else:
                state = policynet.get_state(p_f_x, cnt)
                action, log_pact, _ = policynet.select_action(state)

            action_batch.append(action) 
            logp_batch.append(log_pact)  

        rewards, logp_bs, scalars = reprocess_withmask(action_batch, rank_batch, logp_batch, device, args)

        if mode == 'train':
            if nnoptimizer:
                nnoptimizer.zero_grad()

            scalars = policynet.update_policy(optimizer, rewards, logp_bs, scalars)

            if nnoptimizer :
                print('fintuning')
                clip_grad_norm_([p for p in aa.model.parameters() if p.requires_grad], 3.0)
                nnoptimizer.step()

            if args['ft_tag']:
                aa.tag_inference()
                #print('w: {:.3f}, b: {:.3f}, lmd: {:.3f}'.format(aa.tagweight.item(), aa.tagbias.item(), aa.lmda.item()))
                #writer.add_scalar('tagmodel/weight', aa.tagweight.item(), episode) #*args['batch_size'])
                #writer.add_scalar('tagmodel/bias', aa.tagbias.item(), episode) #*args['batch_size'])
                writer.add_scalar('tagmodel/lmda', aa.lmda.item(), episode) #*args['batch_size'])
                writer.add_scalar('tagmodel/weight', aa.tagweight.data.norm(), episode) #*args['batch_size'])
                writer.add_scalar('tagmodel/bias', aa.tagbias.data.norm(), episode) #*args['batch_size'])
            if args['ft_emb']:
                writer.add_scalar('tagmodel/embweight', aa.embedweight.data.norm(), episode) #*args['batch_size'])
                if args['ft_rnn']:
                    writer.add_scalar('rnn-parameter/rnn_param_norm', compute_param_norm(aa.model), episode)
                    writer.add_scalar('rnn-parameter/rnn_grad_norm', compute_grad_norm(aa.model), episode)


        if writer is not None:
            for name, value in scalars:
                writer.add_scalar(mode + name, value, episode) #*args['batch_size'])

        if episode%args['print_every'] ==0:
            print(mode)
            print('Step: {:,} '.format(episode*args['batch_size']) +
                  ' '.join(['{} = {:.3f}'.format(name, value) for name, value in scalars]))

        if episode%args['save_every'] ==0:
            torch.save( aa.state_dict(), args['checkpoint_dir'] + '/' + args['flavor']+'_aa.pt' )
            save_path = save_checkpoint(policynet.model, optimizer, episode, episode*args['batch_size'], dict(scalars)['/suc_rate'], args, prev_save_path=save_path)
            #save_path = save_checkpoint(model, optimizer, epoch, iter_count, auc05, config, prev_save_path=save_path)



if __name__  =='__main__':
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
    if args['ft_tag'] or args['tag_pretrain'] == True:
        args['taginfer'] = True
    if args['ft_rnn']  == True:
        args['ft_emb'] = True
    print(args)

    main(args)


