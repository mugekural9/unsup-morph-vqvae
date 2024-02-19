from collections import defaultdict
train_lemmas = defaultdict(lambda: 0)
train_tags = defaultdict(lambda: 0)

test_lemmas = defaultdict(lambda: 0)
test_tags  = defaultdict(lambda: 0)
test_golds  = defaultdict(lambda: 0)
train = dict()
test= dict()
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train_verbs", "r") as reader:
    for line in reader:
        lemma, inflected, tags = line.strip().split('\t')
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


both_overlap = 0
lemma_overlap = 0
tags_overlap = 0
neither_overlap = 0
for lemma, inflected, tags in test:
    if lemma in train_lemmas and tags in train_tags:
        both_overlap+=1
    elif lemma in train_lemmas and tags not in train_tags:
        lemma_overlap += 1
    elif lemma not in train_lemmas and tags in train_tags:
        tags_overlap += 1
    elif lemma not in train_lemmas and tags not in train_tags:
        neither_overlap +=1

