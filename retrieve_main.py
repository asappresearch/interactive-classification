import os
from typing import Dict
from pprint import pformat, pprint
import argparse

from tqdm import trange, tqdm
import numpy as np
from collections import defaultdict
from tensorboardX import SummaryWriter
import torch
import torch.nn.functional as F
from torch.optim import Adam
# from torch.optim.optimizer import Optimizer
import random 
from data.embedders.batch_embedder import IndexBatchEmbedder
from data.embedders.fasttext_embedder import FastTextEmbedder

from functools import partial

from utils.config import _get_parser
from utils.utils import recall
from utils.helpers import compute_grad_norm, compute_param_norm, \
    load_checkpoint, get_params, noam_step, parameter_count, repeat_negatives, save_checkpoint


def main(config: Dict):
    pprint(config)
    print('Loading Preprocessor')
    if config['bert']:
        from data.berttokenizer import BTTokenizer, BertBatcher
        from module.bert_trainer import run_epoch
        from module.bertmodel import BertRetrieval
        from transformers import AdamW, WarmupLinearSchedule
        print('loading bert tokenizer')
        preprocessor = BTTokenizer(config) #SplitPreprocessor()
        PAD_ID = preprocessor.padding_idx()
        batch_embedder = BertBatcher(cuda=config['cuda'], pad=PAD_ID)
    else:
        from data.tokenizer import Tokenizer_nltk
        from module.trainer import run_epoch
        from module.model import FAQRetrieval as Retrieval
        preprocessor = Tokenizer_nltk() #SplitPreprocessor()

    # ================================ Load data ================================
    bird_domain = False
    if config['domain'] == 'faq':
        from data.faq_loader import load_data
    elif config['domain'] == 'health':
        from data.health_loader import load_data
    elif config['domain'] == 'bird':
        from data.bird_loader import load_data
        from bdmodule.BUDmodel import Retrieval as Retrieval
        from bdmodule.BUDmodel import Retrieval as BertRetrieval
        bird_domain = True

    train_data, val_data, test_datalist, word_to_index = load_data(config, preprocessor)
    aucresult = defaultdict()
    recallresult = defaultdict()
    # ================================ setting up training environment ================================
    # Set up Tensorboard writer
    writer = SummaryWriter(os.path.join(config['tensorboard_dir'], config['flavor']))
    writer.add_text('Config', pformat(config))

    if not config['bert']:
        print('Loading FastText')
        embedder = FastTextEmbedder(path=config['embedding_path'])
        print('Loading embeddings')
        batch_embedder = IndexBatchEmbedder(embedder, word_to_index, cuda=config['cuda'])

    # Load or build model
    if config['checkpoint_path']:
        print('Loading model from {}'.format(config['checkpoint_path']))
        model, optimizer, init_epoch, iter_count, best_auc05 = load_checkpoint(config['checkpoint_path'])
        save_path = config['checkpoint_path']
    else:
        print('Building model')
        if config['bert']:
            model = BertRetrieval(config)
            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config['weight_decay']},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]
            # optimizer_grouped_parameters = get_params(model)
            optimizer = AdamW(optimizer_grouped_parameters, lr=config['lr'], eps=1e-8)
            num_batch_per_epoch = min(train_data.num_batches, config['max_batches_per_epoch'])
            t_total = int(num_batch_per_epoch // config['gradient_accumulation_steps'] * config['max_epoch'])
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config['warmup_steps'], t_total=t_total)
        else:
            model = Retrieval(config)
            optimizer = Adam(get_params(model),
                            lr=config['lr'], betas=[config['adam_beta1'], config['adam_beta2']], eps=config['adam_eps'])
            scheduler = None
        init_epoch = iter_count = best_auc05 = 0
        save_path = None

    
    # Print model details
    # print(model)
    print('Number of trainable parameters = {:,}'.format(parameter_count(model)))

    model = model.to(model.device)

    # ================================ Training  ================================
    # Run training
    for epoch in trange(init_epoch, config['max_epoch']):
        print('=' * 20 + ' Epoch {} '.format(epoch) + '=' * 20)

        # Train
        iter_count = run_epoch(train_data, batch_embedder, model, config, train=True, iter_count=iter_count,
                                writer=writer, scheduler=scheduler, bird=bird_domain, optimizer=optimizer)
        torch.cuda.empty_cache()

        auc05 = run_epoch(val_data, batch_embedder, model, config, train=False, iter_count=iter_count, writer=writer, bird=bird_domain)

        torch.cuda.empty_cache()

        # Save if improved validation auc@0.05
        # if epoch == 0  or auc05 > best_auc05:
        if epoch%4==0:
            best_auc05 = auc05
            save_path = save_checkpoint(model, optimizer, epoch, iter_count, auc05, config, prev_save_path=save_path)
            for key, test_data in test_datalist.items():
                print('Testing:')
                auc05_test = run_epoch(test_data, batch_embedder, model, config, train=False, test=True,  iter_count=iter_count, writer=writer, bird=bird_domain)
                aucresult[key] = auc05_test
                for key, test_data in test_datalist.items():
                    print('test dataset : {}'.format(key))
                    ks=[1,3,5]
                    recalls = recall(test_data, batch_embedder, model, config['eval_batch_size'], ks=[1,3,5])
                    recallresult[key] = recalls 
                    writer.add_scalar('auc05s/'+key, auc05, iter_count * train_data.batch_size)
                    for i in range(len(recalls)):
                        writer.add_scalar('recall/R' + str(ks[i])+'_'+key, recalls[i], iter_count * train_data.batch_size)


    # Wrap up
    fout  = open('results/'+config['flavor']+'.txt', 'w')
    fout.write(str(config))
    print('Training complete. Best model saved to {}'.format(save_path))
    print('\nauc result: ')
    for k, v in aucresult.items():
        print('{} , {} '.format(k,v))
        fout.write('{} , {}\n'.format(k,v))
    print('\nrecall:')
    for k, v in recallresult.items():
        print(','.join([k] + [str(x) for x in v]))
        fout.write(','.join([k] + [str(x) for x in v]))
        fout.write('\n')
    writer.close()

    # Move model to cpu and prepare for inference
    model.cpu()
    model.eval()
    fout.close()

if __name__ == '__main__':
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

    np.random.seed(args['seed'])
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args['seed'])

    main(args)
