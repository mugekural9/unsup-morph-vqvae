import math
import pandas as pd
from collections import defaultdict
train_lemmas = defaultdict(lambda: 0)
train_tags = defaultdict(lambda: 0)
train_inflected = defaultdict(lambda: 0)

results = pd.read_csv("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tst_results_210.csv")

test_lemmas = defaultdict(lambda: 0)
test_tags  = defaultdict(lambda: 0)
test_golds  = defaultdict(lambda: 0)
train = dict()
test= dict()
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train_verbs", "r") as reader:
    for line in reader:
        lemma, inflected, tags = line.strip().split('\t')
        train_inflected[inflected] += 1

        train_lemmas[lemma] += 1
        train_tags[tags]+=1
        train[(lemma,inflected,tags)] = 0
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS", "r") as reader:
    for line in reader:
        lemma, inflected, tags = line.strip().split('\t')
        test_lemmas[lemma] += 1
        test_tags[tags]+=1
        test_golds[inflected] = tags
        test[(lemma,inflected,tags)] = 0


df = pd.DataFrame({ 'true/false':[], 'lemma':[], 'tags':[], 'seen_in_trn': [], 'number_of_lemma_seen':[], 'number_of_tags_seen':[], 'REINF_TAGFINDER2_pred':[], 'gold':[], 'false_pred_in_TRN':[]})
i = 0
df['lemma'] = df['lemma'].astype(str)
df['tags'] = df['tags'].astype(str)
df['seen_in_trn'] = df['seen_in_trn'].astype(str)
df['gold'] = df['gold'].astype(str)
df['REINF_TAGFINDER2_pred'] = df['REINF_TAGFINDER2_pred'].astype(str)
df['false_pred_in_TRN'] = df['false_pred_in_TRN'].astype(str)

for idx,row in results.iterrows():
    if row['REINF_TAGFINDER1_or_2_em'] == 1.0 or math.isnan(row['REINF_TAGFINDER1_or_2_em']):
        df.at[i, 'true/false'] = 'True'
        df.at[i, 'lemma'] = row['lemma'][:-4]
        gold= row['gold'][:-4]
        tags = test_golds[gold]
        df.at[i, 'tags'] = tags
        df.at[i, 'number_of_lemma_seen'] = train_lemmas[lemma]
        if test_golds[gold] in train_tags:
            df.at[i, 'seen_in_trn'] = 'YES'
            df.at[i, 'number_of_tags_seen'] = train_tags[tags]
        df.at[i,'REINF_TAGFINDER2_pred']= row['REINF_TAGFINDER2_pred'][:-4]
        df.at[i,'gold']=gold
        df.at[i, 'false_pred_in_TRN'] =  str(row['REINF_TAGFINDER2_pred'][:-4] in train_inflected)
        i+=1
df.to_csv('error_analysis_TRUES_210.csv')


#171 mistakes, 108 never seen features, 63 seen -> %63
#171 newly generated false words.
#1275 correct, 607 never seen, 668 seen


