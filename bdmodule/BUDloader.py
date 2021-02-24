import os

from tqdm import tqdm

from pathlib import Path
from collections import defaultdict
import sys
import pandas as pd
import random
import numpy as np
import torch


# np.random.seed(345)
# random.seed(345)
# torch.manual_seed(7)
# if torch.cuda.is_available(): torch.cuda.manual_seed_all(7)
home = str(Path.home())
datadir = home + '/workspace/CUB_data/'

def readcaptions( config=None ):
    home = str(Path.home())
    fnames = os.listdir(datadir+'captions')  
    #fnames = os.listdir('../CUB_data/small_captions')  
    fnames = [datadir+'captions/'+x for x in fnames]
    #fnames = np.random.choice(fnames, 1000)

    capdata = []
    for fnn in fnames:
         with open(fnn) as f:
            fn =fnn.split('/')[-1]   
            cl=' '.join([x.lower() for x in fn.split('_') if not x[0].isdigit()])
            #imageid='_'.join([x.replace('.txt', '').lower() for x in fn.split('_') if x[0].isdigit()])
            imageid = fn.replace('.txt', '')
            #print(imageid)
            cap = [x.strip().lower() for x in f.readlines()]
            sample ={}
            sample['querylist'] = cap
            sample['imageid'] = imageid
            sample['clsname'] = cl
            capdata.append(sample)

    captiondata = pd.DataFrame.from_records(capdata)
    return captiondata


def readtags( config=None ):
    home = str(Path.home())
    attnames = open(datadir+'attributes/mod_att.txt').readlines() ## read the ids for attribute
    tagdic = {}
    for att in attnames:
        tagkey = att.split(' ')[0]
        tag = ' '.join(att.split(' ')[1:]).strip()
        tagdic[tagkey] = tag

    imgnames = open(datadir+'attributes/images.txt').readlines()  ## read the folder and ids for images
    imgdic = {}
    clsname = []
    for img in imgnames:
        imgkey = img.split(' ')[0]
        imgname = img.split(' ')[-1].strip().split('/')[-1].replace('.jpg','')
        imgdic[imgkey] = imgname
        clss  = ' '.join(imgname.lower().split('_')[:-2])
        if clss not in clsname:
            clsname.append(clss)

    certainty_dict={'1':['1','2','3','4'],
                    '2':['2','3','4'],
                    '3':['3','4'],
                    '4':['4']}

    labels = open(datadir+'attributes/image_attribute_labels.txt').readlines() ## perimage annotation 
    img_tag =defaultdict(list)
    for ll in labels:
        ll = ll.strip().split()[:-1]
        #if ll[2] == '1' and ll[3] != '0':
        if ll[2] == '1' and ll[3] in certainty_dict[config['certain_level']]:
            img_tag[imgdic[ll[0]]].append(tagdic[ll[1]])
    tagdata = pd.DataFrame({'imageid':list(img_tag.keys()), 'taglist':list(img_tag.values())})
    #tagdata['clsname'] = tagdata.apply(lambda x:  ' '.join(x.imageid.lower().split('_')[:-2]), axis = 1)


    scale=3
    clsatt = open(datadir+'attributes/class_attribute_labels_continuous.txt').readlines() ## perclass annotation 
    clsatt=[[float(nm) for nm in x.split(' ')] for x in clsatt]
    clsatt = np.array(clsatt)
    clstag_table=np.zeros_like(clsatt)
    clstag_table[np.where(clsatt>clsatt.mean()*scale)] = 1

    cls_att={}
    for i in range(len(clsname)):
        att = [tagdic[str(j+1)] for j in range(clstag_table.shape[1]) if clstag_table[i][j]==1]
        cls_att[clsname[i]] = att

    return tagdata, cls_att



def read_data(  config=None ):
    queryfile = readcaptions(config)
    tagfile, clsatt = readtags(config)
    # ====== merge data and do sampling ======
    alldata = queryfile.join(tagfile.set_index('imageid'), on='imageid', how='inner')

    if config['sampleN']!=0:  ## take 0 as default
        sampleN = config['sampleN']
        goodid = [x.strip() for x in open(f'goodid_{sampleN}.txt', 'r').readlines()]
        alldata = alldata[alldata.imageid.isin(goodid)]
    if config['data_clean'] == 'xs': ## default as full
        good_bd_cls = [x.strip() for x in open('good_bd_clsname.txt', 'r').readlines()]
        alldata = alldata[alldata.clsname.isin(good_bd_cls)]
    print(f'TOTAL Example: {len(alldata)}')

    alldata = alldata.to_dict('records') # got fields of ['clsname', 'imageid', 'querylist', 'taglist']
    return  np.array(alldata), clsatt




def read_protodata(config):

    queryfile = readcaptions(config)
    tagfile, clsatt = readtags(config)
    alldata = queryfile.join(tagfile.set_index('imageid'), on='imageid', how='inner')


    if config['data_clean'] == 'xs':
        good_bd_cls = [x.strip() for x in open('good_bd_clsname.txt', 'r').readlines()]
        alldata = alldata[alldata.clsname.isin(good_bd_cls)]
        
    # ====== processing prototypes ======
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min

    def get_att_dict():
        attnames = open(datadir+'attributes/mod_att.txt').readlines()
        att2id = {}
        for att in attnames:
            #tagkey = att.split(' ')[0]
            tag = ' '.join(att.split(' ')[1:]).strip()
            att2id[tag] = len(att2id)
        id2att ={v:k for k,v in att2id.items()}
        return att2id, id2att
    def taglist_tovec(taglist):
            #taglist = img.taglist
            vec = np.zeros(len(att2id))
            tgids = [att2id[t] for t in taglist]
            vec[np.array(tgids)] = 1
            return vec
    def a_b(a,b):
        return sum(map(lambda a:abs(a[0]-a[1]), zip(a,b)))
    att2id, id2att = get_att_dict()

    newdf = []
    protodf=[]
    cln = config['n_proto']
    for cls,examples in alldata.groupby(['clsname']):
        test = examples
        test['vector'] = test.apply(lambda x: taglist_tovec(x.taglist), axis=1)
        data =  list(test.vector.values )
        km = KMeans(n_clusters=cln).fit(data)

        prototypes, dist = pairwise_distances_argmin_min(km.cluster_centers_, data)
        centerdf = test.iloc[prototypes]
        centerdf['clu_n'] = pd.Series(np.arange(cln),  index=centerdf.index)
        centerdf['proto_text'] = centerdf.apply(lambda x: ', '.join(np.random.choice(x.querylist, 5)), axis=1)

        test['cluster_n'] = km.labels_
        test['proto_text'] = test.apply(lambda x: centerdf.iloc[x.cluster_n].proto_text, axis=1)
        
        newdf.append(test)
        protodf.append(centerdf)
        #test['']
    newdf = pd.concat(newdf)
    protodf = pd.concat(protodf)

    # ====== processing prototypes ======
    alldata = newdf.to_dict('records')
    return  np.array(alldata), clsatt, protodf



def parse_traindata( data, clsatt, config=None,  includetag=False, repeatN=50):
    src =[]
    tgt=[]
    for i, dt in enumerate(data): 

        tagsizes = np.random.choice( np.arange(0, len(dt['taglist'])), repeatN)

        for tagsize in tagsizes: 
            s =[]
            if np.random.binomial(1,0.8): 
                initq = random.choice(dt['querylist'])
                s.append(initq)
            s = " , ".join(list(np.random.choice(dt['taglist'], size=tagsize, replace=False)) + s)
            

            if not includetag:
                t =  dt['clsname']
            else: 
                t = ' , '.join(clsatt[dt['clsname']])
            if config['use_proto']:
                t = dt['proto_text']
            if len(s) == 0 :
                continue
            src.append(s)
            tgt.append(t)
    return np.array(src), np.array(tgt)



def parse_train_perimage( data, clsatt, config=None,  includetag=False, repeatN=50):
    src =[]
    tgt=[]

    for i, dt in enumerate(data): 

        tagsizes = np.random.choice( np.arange(0, len(dt['taglist'])), repeatN)

        queryidx = np.random.choice(len(dt['querylist']), 5, replace=False)
        querypool =[dt['querylist'][i] for i in queryidx]
        imagerep =[dt['querylist'][i] for i in range(len(dt['querylist'])) if i not in queryidx ]
        t = ' <tk> '.join(imagerep)

        for tagsize in tagsizes: 
            s =[]
            if np.random.binomial(1,0.8): 
                initq = random.choice(querypool)
                s.append(initq)
            s = " , ".join(list(np.random.choice(dt['taglist'], size=tagsize, replace=False)) + s)
            if len(s) == 0 :
                continue
            src.append(s)
            tgt.append(t)
    return np.array(src), np.array(tgt)




def parse_valdata( data, clsatt, initialq, tag_r , config, includetag=False, sampleN=5):
    src =[]
    tgt=[]

    for i, dt in enumerate(data): 

            if not includetag:
                t =  dt['clsname']
            else: 
                t = ', '.join(clsatt[dt['clsname']])
            if config['use_proto']:
                t = dt['proto_text']

            for k in range(sampleN):
                s =[]

                if initialq: 
                    s.append(random.choice(dt['querylist']))
                #tagsize = max(1, int(len(dt['taglist']) * tag_r))
                tagsize = int(len(dt['taglist']) * tag_r)
                s =  " , ".join(s + list(np.random.choice(dt['taglist'], size=tagsize, replace=False)))
                if len(s) == 0 :
                    continue
                src.append(s)
                tgt.append(t)
    return np.array(src), np.array(tgt)




def parse_val_perimage( data, clsatt, initialq, tag_r , includetag=False, sampleN=5):
    src =[]
    tgt=[]

    for i, dt in enumerate(data): 
        queryidx = np.random.choice(len(dt['querylist']), 5, replace=False)
        querypool =[dt['querylist'][i] for i in queryidx]
        imagerep =[dt['querylist'][i] for i in range(len(dt['querylist'])) if i not in queryidx ]
        t = ' <tk> '.join(imagerep)
        for k in range(sampleN):
            s =[]

            if initialq: 
                s.append(random.choice(querypool))
            tagsize = max(0, int(len(dt['taglist']) * tag_r))
            s =  " , ".join(list(np.random.choice(dt['taglist'], size=tagsize, replace=False)) + s)
            if len(s) == 0 :
                continue
            src.append(s)
            tgt.append(t)
    return np.array(src), np.array(tgt)




def split_srctgtdata(src, tgt, r = 0.15):
    #src , tgt = np.array(src), np.array(tgt)
    Ntot = len(src)
    perm = np.random.permutation(Ntot)
    train_index  = perm[int(Ntot*r):]
    val_index = perm[: int(Ntot*r)]

    src_train, tgt_train = src[train_index], tgt[train_index]
    src_val, tgt_val = src[val_index], tgt[val_index]

    print('There are {} training examples'.format(len(src_train)))
    print('There are {} validation examples'.format(len(src_val)))

    return src_train, tgt_train, src_val, tgt_val



def split_data(records, r = 0.15):
    Ntot = len(records)
    perm = np.random.permutation(Ntot)
    train_index  = perm[int(Ntot*r):]
    val_index = perm[: int(Ntot*r)]

    data_train = records[train_index]
    data_val = records[val_index]
    return data_train, data_val