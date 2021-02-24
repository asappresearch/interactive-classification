import argparse
from tqdm import tqdm, trange
import os
from typing import *
import pdb
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn.functional import normalize


from utils.helpers import compute_grad_norm, compute_param_norm, \
    load_checkpoint, get_params, noam_step, parameter_count, repeat_negatives, save_checkpoint


class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.device = torch.device('cuda') if args['cuda'] else torch.device('cpu')
        self.state_space = args['state_n'] if args['state_n'] else 535
        self.state_space += args['max_step']
        self.action_space = 2
        self.dropout = args['policy_dropout']
        self.args = args

        self.model = torch.nn.Sequential(
            nn.Linear(self.state_space,  args['hidden_n']),
            nn.Dropout(p=self.dropout),
            nn.Tanh(),
            nn.Linear( args['hidden_n'], self.action_space),
            nn.Softmax(dim=-1)
        )
        
        self.gamma = args['gamma'] 
        self.model = self.model.to(self.device)


    def forward(self, x):    
        return self.model(x)

    def select_action(self, state):
        #Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        acts = self.forward(state)
        c = Categorical(acts)
        action = c.sample()
        firerate = c.probs[:, 1].cpu().detach().numpy()
        return action.cpu().detach().numpy(), c.log_prob(action), firerate


    def get_state(self, p_f_x, cnt):
        state_fqs = p_f_x.topk(self.args['state_n'])[0] if self.args['state_n'] else p_f_x
        feature_t = torch.zeros(state_fqs.shape[0], self.args['max_step']).to(self.device)
        feature_t[:, cnt-1]=1
        state = torch.cat([state_fqs, feature_t], 1)
        return state

    def update_policy(self, optimizer, rewards, logp_bs, scalars, nnoptimizer=None):

        loss = torch.sum(rewards * torch.sum(logp_bs,1) )*(-1)/ len(rewards)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scalars.append(('-parameter/pnet_grad_norm', compute_grad_norm(self.model)))
        scalars.append(('-parameter/pnet_param_norm', compute_param_norm(self.model)))
        scalars.append(('-parameter/pnet_loss', loss.item()))
    
        del rewards
        del logp_bs
        return scalars
