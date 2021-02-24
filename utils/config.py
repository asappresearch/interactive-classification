from typing import Dict

import argparse

def _get_parser(parser):
    
    # general args
#     parser.add_argument('--datatype', type=str, default='initialquery',  help='Flavor of the AutoSuggest model being trained')
    parser.add_argument('--includetag', type=str, default='False', choices=['False', 'True'], help='if have tags in the faq representation')    
#     parser.add_argument('--nostrutag', type=str, default='False', choices=['False', 'True'], help='if include the auto-generated/key words based tag')
#     parser.add_argument('--toignore', type=str, default='', help='if include the auto-generated/key words based tag')
#     parser.add_argument('--dedupe', type=str, default='False', choices=['False', 'True'], help='if have tags in the faq representation')
    parser.add_argument('--flavor', type=str, default='run0', help='model name')
    parser.add_argument('--para_ratio', type=float, default=0.0, help='the probability of using paraphrased data')

    parser.add_argument('--checkpoint_dir', type=str, default='_checkpoints', help='Directory in which model checkpoints will be saved')
    parser.add_argument('--tensorboard_dir', type=str, default='_tensorboard', help='Directory in which tensorboard logs will be saved')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint if loading a model')
    parser.add_argument('--evaluate_only', type=str, default='False', choices=['False', 'True'], help='Only perform evaluation (no training)')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Flag to turn off cuda (i.e. CPU only)')

    parser.add_argument('--print_frequency', type=int, default=200, help='Number of steps between each printing of batch statistics')
    
    # transformer args
    parser.add_argument('--embedding_path', type=str, default= '../embeddings/fasttext_bin-spear20180618.bin', help='Path to .bin file containing FastText embeddings')
    parser.add_argument('--embedding_type', type=str, default='fasttext', help='Path to .bin file containing FastText embeddings')
    parser.add_argument("--max_context_length", type=int, default=200, help="Maximum number of words in a context")
    parser.add_argument("--max_text_length", type=int, default=50, help="Maximum number of words in a response")
    # model args
    parser.add_argument('--loss_type', type=str, default= 'cross_entropy', help='Loss function used for the model ')
    parser.add_argument('--rnn_type', type=str, default= 'sru', help='Loss function used for the model ')
    parser.add_argument('--grad_clip', type=float, default= 3.0, help='Input word embedding size')
    parser.add_argument('--embedding_size', type=int, default=300, help='Input word embedding size')
    parser.add_argument('--hidden_size', type=int, default=150, help='Hidden size for RNN')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of RNN layers')
    parser.add_argument('--unidirectional', action='store_true', default=False, help='Unidirectional rather than bidirectional RNN')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')
    parser.add_argument('--rnn_dropout', type=float, default=0.0, help='Variational RNN dropout')
    parser.add_argument('--num_attention_units', type=int, default=64, help='Number of attention units')
    parser.add_argument('--num_attention_heads', type=int, default=16, help='Number of attention heads')

    # optimizer args
    parser.add_argument('--warmup_steps', type=int, default=4000, help='Number of warmup steps in noam learning rate scheduler')
    parser.add_argument('--noam_scaling_factor', type=float, default=2.0, help='Scaling factor by which to multiply learning rate computed by noam lr scheduler')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate (note: this is replaced by noam scheduler learning rate)')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='beta1 value for Adam optimizer')
    parser.add_argument('--adam_beta2', type=float, default=0.98, help='beta2 value for Adam optimizer')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='eps value for Adam optimizer')

    # training args
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=120, help='Batch size during evaluation')
    parser.add_argument('--max_epoch', type=int, default=100, help='Maximum number of epochs to train for')
    parser.add_argument('--max_batches_per_epoch', type=int, default=1000000, help='Maximum number of batches in a training epoch')
    parser.add_argument('--num_negatives', type=int, default=100, help='Number of negative responses for each positive response during training'
                             '(Note: these negatives are reused across a batch)')
    parser.add_argument('--eval_num_negatives', type=int, default=100, help='Number of negative responses for each positive response during evaluation'
                             '(Note: these negatives are NOT reused across a batch)')

    # maxinum info parameters parameters
    parser.add_argument('--ckpt', type=str, default='_checkpoints/attold_heads32_dp0.1_nl1_auc05_0.811_e34.pt', help='check point file ')
    parser.add_argument('--embeddertype', type=str, default='index', choices=['word', 'index'], help='if have tags in the faq representation')
    parser.add_argument('--strategy', type=str, default='infogain', choices=['infogain','tag_baseline','simple_no','onestep_info','random'], help ='metrics to report')    
    parser.add_argument("--max_step", type=int, default=10, help="Flag to print additional outputs")

    # tag uncertainty parameters
    parser.add_argument("--taginfer", action='store_true', default=False, help="Flag to print additional outputs")
    parser.add_argument('--tag_threshold', type=float, default=0.02, help='if have tags in the faq representation')
    parser.add_argument('--tagckpt', type=str, default='_checkpoints/tag_pre2_auc05_0.858_e99.pt', help='if have tags in the faq representation')

    # printing/setting parameters
    parser.add_argument('--interactive', action='store_true', default=False, help='if have tags in the faq representation')
    parser.add_argument('--comment', type=str, default='', help='comment to write to the tensorboard')
    parser.add_argument("--debug", action='store_true', default=False, help="Flag to print additional outputs")
    parser.add_argument("--verbose", action='store_true', default=False, help="Flag to print additional outputs")

    parser.add_argument('--zeroshot', action='store_true', default=False, help='if using IG in policy network')
    parser.add_argument('--aackpt', type=str, default='_checkpoints/0215rp_tagmodelscalar_ulmd1.0_r-0.25_tag_pretrainT_fttagTrue_ftrnnTrue_aa.pt', help='if have tags in the faq representation')
    parser.add_argument('--policyckpt', type=str, default='_checkpoints/0215rp_tagmodelscalar_ulmd1.0_r-0.25_tag_pretrainT_fttagTrue_ftrnnTrue_auc05_0.786_e4900.pt', help='if have tags in the faq representation')

    # ========= bird training specific ============
    parser.add_argument('--bd_agent', type=str, default='noise', choices=['base','noise','proto'], help='if have tags in the faq representation')
    parser.add_argument("--sampleN", type=int, default=0, help="using distance to reduce the number of bird image examples ")
    parser.add_argument('--data_clean', type=str, default='full', choices=['full','xs','proto'], help='if have tags in the faq representation')

    parser.add_argument("--certain_level", type=str, default='3',  choices=['1','2','3','4'], help="reward of getting the answer correct")

    parser.add_argument("--no_rpt_ft", action='store_true', default=False, help="reward of getting the answer correct")
    parser.add_argument("--n_proto", type=int, default=3, help="reward of getting the answer correct")
    parser.add_argument("--use_proto", action='store_true', default=False, help="reward of getting the answer correct")
    # parser.add_argument("--single_example", action='store_true', default=False, help=" learning learning_rate ")
    parser.add_argument("--use_bilinear", type=str, default='False', choices=['False','True'], help=" learning learning_rate ")
    parser.add_argument("--train_with_image", type=str, default='False', choices=['False','True'], help="number of hidden dimension ")

    parser.add_argument('--seed', type=int, default=345, help='if have tags in the faq representation')


    # PG parameters
    parser.add_argument("--episodes", type=int, default=5000, help="total episodes to run")
    parser.add_argument("--print_every", type=int, default=5, help="every n episodes print the stats")
    parser.add_argument("--save_every", type=int, default=100, help="Flag to print additional outputs")
    parser.add_argument("--eval_every", type=int, default=2, help="every n episodes print the stats")
    parser.add_argument("--test_every", type=int, default=10, help="every n episodes print the stats")
    
    parser.add_argument("--gamma", type=float, default=1.0, help="Flag to print additional outputs")
    parser.add_argument("--ft_tag", action='store_true', default=False, help='the method ')
    parser.add_argument("--ft_emb", action='store_true', default=False, help='the method ')

    parser.add_argument("--ft_rnn", action='store_true', default=False, help='the method ')

    parser.add_argument("--pfx_factorization", type=str, default='full', help='the method ')

    parser.add_argument("--reward_p",type= float, default=20, help="reward of getting the answer correct")
    parser.add_argument("--reward_n",type= float, default=-10, help="reward of getting the answer wrong")
    parser.add_argument("--reward_ask",type= float, default=-1, help="reward of more turns")
    parser.add_argument("--state_n",type= int, default=20, help="number of state, top n px ")
    parser.add_argument("--hidden_n",type= int, default=32, help="number of hidden dimension ")
    parser.add_argument("--aws", action='store_true', default=False, help="Flag to print additional outputs")
    parser.add_argument("--ft_lr",type= float, default=1e-3, help=" learning learning_rate ")
    parser.add_argument("--ft_episode",type= int, default=0, help=" learning learning_rate ")

    parser.add_argument("--datasplit",type= str, default='faq', help=" learning learning_rate ")
    parser.add_argument("--policy_dropout",type= float, default=0.0 , help=" learning learning_rate ")


    parser.add_argument("--no_para_data", action='store_true', default=False, help="Flag to print additional outputs")
    parser.add_argument("--no_tag_data", action='store_true', default=False, help="Flag to print additional outputs")
    parser.add_argument("--cv_n", type=int, default=2, help="Flag to print additional outputs")
    parser.add_argument("--use_attention", action='store_true', default=False,  help="Flag to print additional outputs")
    parser.add_argument("--tag_faq_separator", type=str, default = '@ ',  help="Flag to print additional outputs")


    parser.add_argument('--belief_reduce', type=float, default=0.0, help='if have tags in the faq representation')
    parser.add_argument('--user_uncertain', type=float, default=0.0, help='if have tags in the faq representation')
    parser.add_argument('--user_type', type=str, default='persona', help='if have tags in the faq representation')
    parser.add_argument('--user_lamda', type=float, default=1.0, help='if have tags in the faq representation')

    parser.add_argument('--tag_model', type=str, default='scalar', choices=['bl','scalar','vector'], help='if have tags in the faq representation')
    parser.add_argument('--tag_pretrain', action='store_true', default=False, help='if have tags in the faq representation')
    parser.add_argument('--aa_ld', type=float, default=0.1, help='if have tags in the faq representation')
    parser.add_argument('--sampled', type=int, default=1, help='if have tags in the faq representation')

    # demo args
    parser.add_argument('--db_path', type=str, default='data.db', help='path to DB for saving interactions')
    parser.add_argument('--save_to_db', action='store_true', default=False, help='Save interactions to database')

    # bert args                    
    parser.add_argument('--bert', action='store_true', default=False, help='using bert model')
    parser.add_argument('--bert_type', type=str, default='bert-base-uncased', help='using bert model')
    parser.add_argument('--bert_pooling', type=str, default='first', choices=['first', 'average'], help='using bert model')
    parser.add_argument('--bert_freeze_embedding', type=str, default='False', choices=['False', 'True'], help='using bert model')
    parser.add_argument('--bert_freeze_encoder', type=str, default='False', choices=['False', 'True'], help='using bert model')
    parser.add_argument('--bert_freeze_all', type=str, default='False', choices=['False', 'True'], help='using bert model')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,  help='using bert model')

    parser.add_argument('--domain', type=str, default='faq', choices=['faq', 'bird', 'health'], help='using bert model')
    parser.add_argument('--inputf', type=str, default='old', choices=['old', 'new'], help='using bert model')
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    
    parser.add_argument('--using_categorical', action='store_true', default=False, help='using bert model')

    return parser