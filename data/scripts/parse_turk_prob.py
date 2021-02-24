import csv
import json
import random
from collections import defaultdict
import numpy as np
import os

#from sklearn.metrics import cohen_kappa_score

mean_pairwise_kappa = 0
total = 0

def kappa(stacked_counts):
    global mean_pairwise_kappa
    global total
    if stacked_counts.shape[0] == 3:
        rater0 = stacked_counts[0]
        rater1 = stacked_counts[1]
        rater2 = stacked_counts[2]
        kappa01 = cohen_kappa_score(rater0, rater1)
        kappa12 = cohen_kappa_score(rater1, rater2)
        kappa02 = cohen_kappa_score(rater0, rater2)
        mean_pairwise_kappa += (kappa01 + kappa12 + kappa02)
        total += 3
    elif stacked_counts.shape[0] == 4:
        rater0 = stacked_counts[0]
        rater1 = stacked_counts[1]
        rater2 = stacked_counts[2]
        rater3 = stacked_counts[3]
        kappa01 = cohen_kappa_score(rater0, rater1)
        kappa02 = cohen_kappa_score(rater0, rater2)
        kappa03 = cohen_kappa_score(rater0, rater3)
        kappa12 = cohen_kappa_score(rater1, rater2)
        kappa13 = cohen_kappa_score(rater1, rater3)
        kappa23 = cohen_kappa_score(rater2, rater3)
        mean_pairwise_kappa += (kappa01 + kappa02 + kappa03 + kappa12 + kappa13 + kappa23)
        total += 6

def parse_turk_data(save=False):
dir_name = '2019-01-batch1'
name = '30_to40'
file_name = 'tag_forannotation1208/allfile_tags_from{}.csv'.format(name)
turker_file_name = os.path.join(dir_name, '{}_results.csv'.format(name))
turk_data = []
original_items = {}

with open(file_name) as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        row = dict(row)
        original_items[row['faq_id']] = [
            row['item1'], row['item2'], row['item3'], row['item4'], row['item5'], 
            row['item6'], row['item7'], row['item8'], row['item9'], row['item10'],
        ]

with open(turker_file_name) as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
        row = dict(row)
        selected_items = row['Answer.faq'].split('|')
        faq_id = row['Input.faq_id']

        inds = [int(s[4:]) - 1 for s in selected_items if s != 'na']
        inds_with_na = [int(s[4:]) - 1 if s != 'na' else 10 for s in selected_items]
        bagofcounts = [0 for _ in range(11)]

        for ind in inds_with_na:
            bagofcounts[ind] = 1

        datapoint = {
            'faq_id': faq_id,
            'answers': [original_items[faq_id][ind] for ind in inds],
            'inds_with_na': inds_with_na,
            'bagofcounts': np.array(bagofcounts),
        }
        turk_data.append(datapoint)

tag_lengths = []
turker_selections = defaultdict(list)
for datapoint in turk_data:
    turker_selections[datapoint['faq_id']].append(datapoint['bagofcounts'])
    tag_lengths.append(len(datapoint['answers']))
print('Mean length:', sum(tag_lengths) / len(tag_lengths))

sampled_faqs = {}
all_faqs = {}
num_na = 0
total_completed = 0
for faq_id, array_list in turker_selections.items():
    items = original_items[faq_id]
    sampled_array_list = random.sample(array_list, 2)
    sampled_stack_counts = np.stack(sampled_array_list, 0)
    sampled_probs = np.sum(sampled_stack_counts[:, :-1], 0) / len(sampled_array_list)

    stacked_counts = np.stack(array_list, 0)
    num_na += np.sum(stacked_counts[:, -1])
    total_completed += len(array_list)
    probs = np.sum(stacked_counts[:, :-1], 0) / len(array_list)

#        kappa(stacked_counts)

    sampled_faqs[faq_id] = {items[i]: sampled_probs[i] for i in range(len(items))}
    all_faqs[faq_id] = {items[i]: probs[i] for i in range(len(items))}
    
print(num_na / total_completed)
#    print('mean pairwise kappa:', mean_pairwise_kappa / total)

if save:
#        with open('sampled2_faq_probs_{}.json'.format(name), 'w') as f:
#            f.write(json.dumps(sampled_faqs))

    with open(os.path.join(dir_name, 'faq_probs_{}.json'.format(name)), 'w') as f:
        f.write(json.dumps(all_faqs))

def merge_files():        
    file_names = [
        'faq_probs_0to10.json', 
        'faq_probs_10to20.json', 
        'faq_probs_20to30.json', 
        'faq_probs_30to40.json'
    ]
    data = defaultdict(dict)
    for file_name in file_names:
        with open(file_name) as f:
            for line in f:
                obj = json.loads(line)
                data[obj['faq_id']] = {**obj['probs'], **data[obj['faq_id']]}

    with open('faq_probs_0to40.json', 'w') as f:
        f.write(json.dumps(data))

if __name__ == '__main__':
    parse_turk_data(save=True)
#    merge_files()
