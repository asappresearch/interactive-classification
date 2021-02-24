import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
import pandas as pd
import json
import bdmodule.bd_utils as bd_utils
#from module.data_loading_fromlocal import read_turkprob


class UserSimulator(ABC):
    @abstractmethod
    def answer(self ):
        pass


class NoisyUser(UserSimulator):
    def __init__(self, args):
        print('using oracle based user. ')
        self.uncertainty = args['user_uncertain']

    def answer(self, best_ft, batch, mode='test'):
        self.qr_fact = batch[1]
        answer = []
        for i in range(len(best_ft)):
            a = self.qr_fact[i, best_ft[i]]
            if np.random.binomial(1, self.uncertainty): 
                #print('lying')
                a = 1 if np.random.binomial(1, 0.5) else 0
            answer.append(a)
        return answer


class Singleexample_User(UserSimulator):
    def __init__(self, args):
        print('using oracle based user. ')
        self.uncertainty = args['user_uncertain']

        label2att, att2label = bd_utils.att2label()

        if args['using_categorical']:
            binary_idx, categorical, tag_new2old, tag_old2new = bd_utils.parse_cat_binary()
            self.label2att = {lb:[tag_old2new[i] for i in label2att[lb]] for lb in label2att}
            self.att2label = [att2label[tag_new2old[new]]  for new in tag_new2old]
        else:
            self.label2att, self.att2label = label2att, att2label

        self.labellist = list(self.label2att.keys())

    def answer_label(self, best_ft, batch, mode='test'):
        self.qr_fact = batch[1]
        answer = []
        answered_ft = []
        for i in range(len(best_ft)):
            ft = best_ft[i]
            if ft in self.qr_fact[i]:
                answer.append(1)
                answered_ft.append([ft])
                continue

            label = self.labellist[self.att2label[ft]]
            related_att = self.label2att[label]
            a=0
            for att in related_att :
                a += int( att in self.qr_fact[i])  
            if a==0:    ## Nothing in the same category
                answer.append(-1)
                answered_ft.append(related_att)
            else:  ## somthing in the category and the one asked is wrong
                answer.append(0)
                answered_ft.append([ft])
        return answer, answered_ft

    def answer_label_cat(self, best_ft, batch, aa, mode='test'):
        self.qr_fact = batch[1]
        self.tgts = batch[2].cpu().numpy()
        answer = []
        answered_ft = []
        for i in range(len(best_ft)):
            ft = best_ft[i]
            if ft < aa.nq_bi:
                if ft in self.qr_fact[i]:
                    answer.append(1)
                    answered_ft.append([ft])
                    continue

                label = self.labellist[self.att2label[ft]]
                related_att = self.label2att[label]
                a=0
                for att in related_att :
                    a += int( att in self.qr_fact[i])  
                if a==0:  
                    answer.append(-1)
                    answered_ft.append(related_att)
                else:
                    answer.append(0)
                    answered_ft.append([ft])
            else:
                cat = aa.categorical[ ft-aa.nq_bi ]['idx']
                overlap = set(cat) & set(self.qr_fact[i])
                if len(overlap) ==0:
                    a = -1
                else:
                    a = np.random.choice(list(overlap))
                answer.append(a)
                answered_ft.append([ft])
        assert len(answer) == len(answered_ft)

        return answer, answered_ft

    def answer(self, best_ft, batch, mode='test'):

        self.qr_fact = batch[1]
        answer = []
        answered_ft = []
        for i in range(len(best_ft)):
            ft = best_ft[i]
            if ft in self.qr_fact[i]:
                a = 1
            else:
                a =0
            answer.append(a)
        return answer


# class PersonaUser(UserSimulator):

    # def __init__(self, aa, args):
    #     print('using persona based user. ')
    #     self.args = args
    #     self.lamda  = 1 #args['user_lamda']
    #     self.uncertainty = args['user_uncertain']
    #     self.init_user_fromprob(aa)
    #     #self.init_user_data(aa)



    # def init_user_fromprob(self, aa):

    #     faq_probs = read_turkprob('full/')
    #     #faq_probs = read_turkprob('sampled/sampled_')
    #     prob_weight = [1, 1, 1, 1, 1]

    #     query = aa.queryfile
    #     datarecords= query.to_dict('records')
    #     fq_tag_user = np.zeros(aa.gold_table.shape)
        
    #     for i in range(len( datarecords)):
    #         dr = datarecords[i]

    #         tgttext = dr['faqtext'] if 'faqtext' in dr else dr['faq_original']
    #         tgt_ids = aa.faqs.index(tgttext)
    #         if i != tgt_ids:
    #             print(i)
    #         faqid = str(dr['faq_id'])
            
            
    #         labeled = [int(faqid in fp) for fp in faq_probs]
    #         if not 0 in labeled: 
    #             for i in range(len(faq_probs)):
    #                 probdict = faq_probs[i][faqid]
    #                 for tg in probdict.keys():
    #                     if tg in aa.tag_w2i:
    #                         fq_tag_user[tgt_ids, aa.tag_w2i[tg]] =  probdict[tg]* prob_weight[i]
    #                         #fq_tag_user[tgt_ids, aa.tag_w2i[tg]] =  int(probdict[tg] >=0.4)
    #         else:
    #             print('no data')
    #             taglist = dr['taglist']
    #             for tg in taglist:
    #                 if tg in aa.tag_w2i:
    #                     fq_tag_user[tgt_ids, aa.tag_w2i[tg]] =  1

    #     goldtable = aa.gold_table
    #     self.qtag_belief =  self.lamda *fq_tag_user + (1- self.lamda )*goldtable



    # def answer(self, best_ft, batch, mode='test'):
    #     self.tgts = batch[2].cpu().numpy()
    #     answer = []
    #     for i in range(len(best_ft)):
    #         pa = min(1, self.qtag_belief[ self.tgts[i], best_ft[i]])
            
    #         a = 1  if np.random.binomial(1, pa) else 0
    #         '''
    #         if mode == 'train':
    #             a = int(pa >0.4)
    #         else:
    #             a = 1  if np.random.binomial(1, pa) else 0
    #         '''
    #         #if np.random.binomial(1, self.uncertainty): 
    #         #    #print('lying')
    #         #    a = 1 if np.random.binomial(1, 0.5) else 0
    #         answer.append(a)
    #     return answer




    # # def readfold_cvs(self, path ):
    # #         #fold = self.args['cv_n']
    # #         #path = 'paf_anno_result/tag_anno_{}_result.csv'.format(fold)
    # #         tagfile = pd.read_csv(path)
    # #         tagfile = tagfile.drop(['HITId', 'HITTypeId', 'Title', 'Description', 'Keywords', 'Reward',
    # #                'CreationTime', 'MaxAssignments', 'RequesterAnnotation',
    # #                'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds',
    # #                'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds',
    # #                'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime',
    # #                'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime',
    # #                'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate',
    # #                'Last30DaysApprovalRate', 'Last7DaysApprovalRate', ], 1)
    # #         tagfile['pos_tag'] = tagfile.apply(lambda x: [x['Input.'+key] if key!='na' else None for key in x['Answer.faq'].split('|')], 1) 
    # #         tags= tagfile.groupby(['Input.faq_id']).apply(lambda x: [tag for cnt in x['pos_tag'].tolist() for tag in cnt]).reset_index()
    # #         tags.rename(columns = {0:'all_pos_tag'}, inplace = True)
    # #         turk_cnt = tagfile.groupby(['Input.faq_id']).apply(lambda x: len(x)).reset_index()
    # #         turk_cnt.rename(columns = {0:'turk_cnt'}, inplace = True)
    # #         tags = tags.join(turk_cnt.set_index('Input.faq_id'), on='Input.faq_id', how='outer')
    # #         tags = tags.rename(columns={'Input.faq_id':'faq_id'})
    # #         return tags







    # # def init_user_data(self, aa):
    # #     #================Reading and merging files ================
    # #     alltags = []
    # #     for i in range(5):
    # #         path = 'paf_anno_result/tag_anno_{}_result.csv'.format(str(i))
    # #         alltags.append(self.readfold(path))
    # #     tags_0_5 = pd.concat(alltags)
        
    # #     alltags = []
    # #     for i in range(5):
    # #         path = 'paf_anno_result/tag_from5_to15_file{}_result.csv'.format(str(i))
    # #         #print(path)
    # #         alltags.append(self.readfold(path))
    # #     tags_5_15 = pd.concat(alltags)
        
    # #     tag0_15 = tags_0_5.join(tags_5_15.set_index('faq_id'), on='faq_id',lsuffix='_5', rsuffix='_15', how='outer')
    # #     tag0_15['all_pos_tag'] = tag0_15['all_pos_tag_5'] +tag0_15['all_pos_tag_15']
    # #     tag0_15['turk_cnt'] = (tag0_15['turk_cnt_5'] +tag0_15['turk_cnt_15'])/2
        
    # #     tags=tag0_15

    # #     #================Build the belief table  ================
    # #     data_test = aa.queryfile
    # #     test_tag = data_test.join(tags.set_index('faq_id'), on='faq_id', how='outer').replace(np.nan, 0, regex=True)
    # #     test_tag_record = test_tag.to_dict('records')
    # #     aa_to_user_idx={}
    # #     fq_tag=[]
    # #     tgt_in_aa = []
    # #     for i in range(len( test_tag_record )):
    # #         dr = test_tag_record[i]
    # #         tgttext = dr['faqtext'] if 'faqtext' in dr else dr['faq_original']
    # #         tgt_ids = aa.faqs.index(tgttext)
    # #         tgt_in_aa.append(tgt_ids)
    # #         aa_to_user_idx[tgt_ids] = i
    # #         tag_cnt = [0]*len(aa.tag_w2i)

    # #         turk_cnt = dr['turk_cnt']
    # #         dr['all_pos_tag'] = [] if dr['all_pos_tag']== 0 else dr['all_pos_tag']
    # #         while turk_cnt<3:   
    # #             dr['all_pos_tag'] +=   dr['taglist']           
    # #             turk_cnt +=1
    # #             print('for faq :{}, get tags from gold table {}'.format(dr['faq_id'], dr['taglist']))

    # #         for tg in dr['all_pos_tag']:
    # #             if tg in aa.tag_w2i:
    # #                 tag_cnt[ aa.tag_w2i[tg]] = min(3, tag_cnt[ aa.tag_w2i[tg]] +1)
    # #         fq_tag.append(tag_cnt)
    # #     fq_tag = np.array(fq_tag)/3
    # #     goldtable = aa.gold_table[tgt_in_aa]


    # #     self.aa_to_user_idx = aa_to_user_idx
    # #     self.qtag_belief =  self.lamda *fq_tag + (1- self.lamda )*goldtable


