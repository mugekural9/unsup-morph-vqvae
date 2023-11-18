from collections import defaultdict
from dataset.morph_dataset import MorphDataset
import pandas as pd
## trn
trainset = MorphDataset("dataset/tur_large.train")
trn_unique_lemmas_df = pd.DataFrame({'lemma':[], 'count':[]})
trn_unique_tgts_df = pd.DataFrame({'tgt':[], 'count':[]})
trn_unique_lemmas_dict = defaultdict(lambda:0)
trn_unique_tgts_dict =  defaultdict(lambda:0)
for lemma in trainset.lemmas:
    trn_unique_lemmas_dict[lemma] +=1
for tgt in trainset.tgts:
    trn_unique_tgts_dict[tgt] +=1
idx =0
for k,v in trn_unique_lemmas_dict.items():
    trn_unique_lemmas_df.at[idx,'lemma'] = k
    trn_unique_lemmas_df.at[idx,'count'] = v
    idx+=1
idx =0
for k,v in trn_unique_tgts_dict.items():
    trn_unique_tgts_df.at[idx,'tgt'] = k
    trn_unique_tgts_df.at[idx,'count'] = v
    idx+=1
trn_unique_lemmas_df['count'] = trn_unique_lemmas_df['count'].astype(int)
trn_unique_tgts_df['count'] = trn_unique_tgts_df['count'].astype(int)
trn_unique_lemmas_df.to_csv('dataset/trn_lemmas.csv')
trn_unique_tgts_df.to_csv('dataset/trn_tgts.csv')


##dev
devset = MorphDataset("dataset/tur.dev", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
dev_unique_lemmas_df = pd.DataFrame({'lemma':[], 'count':[]})
dev_unique_tgts_df = pd.DataFrame({'tgt':[], 'count':[]})
dev_unique_lemmas_dict = defaultdict(lambda:0)
dev_unique_tgts_dict =  defaultdict(lambda:0)
for lemma in devset.lemmas:
    dev_unique_lemmas_dict[lemma] +=1
for tgt in devset.tgts:
    dev_unique_tgts_dict[tgt] +=1
idx =0
for k,v in dev_unique_lemmas_dict.items():
    dev_unique_lemmas_df.at[idx,'lemma'] = k
    dev_unique_lemmas_df.at[idx,'count'] = v
    idx+=1
idx =0
for k,v in dev_unique_tgts_dict.items():
    dev_unique_tgts_df.at[idx,'tgt'] = k
    dev_unique_tgts_df.at[idx,'count'] = v
    idx+=1
dev_unique_lemmas_df['count'] = dev_unique_lemmas_df['count'].astype(int)
dev_unique_tgts_df['count'] = dev_unique_tgts_df['count'].astype(int)
dev_unique_lemmas_df.to_csv('dataset/dev_lemmas.csv')
dev_unique_tgts_df.to_csv('dataset/dev_tgts.csv')


idx=0
lemma_seen_in_trn = pd.DataFrame({})
for lemma,v in dev_unique_lemmas_dict.items():
    if lemma in trn_unique_lemmas_dict:
        lemma_seen_in_trn.at[idx, 'lemma'] = lemma
        lemma_seen_in_trn.at[idx, 'count'] = trn_unique_lemmas_dict[lemma]
        idx+=1
lemma_seen_in_trn.to_csv('dataset/dev_lemma_seen_in_trn.csv')

idx=0
tgt_seen_in_trn = pd.DataFrame({})
for tgt,v in dev_unique_tgts_dict.items():
    if tgt in trn_unique_tgts_dict:
        tgt_seen_in_trn.at[idx, 'tgt'] = tgt
        tgt_seen_in_trn.at[idx, 'count'] = trn_unique_tgts_dict[tgt]
        idx+=1
tgt_seen_in_trn.to_csv('dataset/dev_tgt_seen_in_trn.csv')



