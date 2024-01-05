import json
from collections import defaultdict
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/results/vqvae/unsup/v2_BITrue_2x10_UtrNOUN_zLEM100_zTAG128_decnh1024_kl1.0_epc10_strt5_decdo0.2_inpemb128_bsize64/qinds_tst_epc47.json") as json_file:
    data = json.load(json_file)

tags_2_entries = dict()
word_tags = dict()
nounfile = '/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_NOUNS'
with open(nounfile) as reader:
    for line in reader:
        line = line.strip()
        lemma, tgt, tags = line.split('\t')
        word_tags[tgt] = tags
        tags_2_entries[tags] = defaultdict(lambda:0)

for key,val in data.items():
    #key is a dict entry
    for word in val:
        tag = word_tags[word[:-4]]
        tags_2_entries[tag][key] +=1


with open("tag_to_entry.json", "w") as outfile:
    json.dump(tags_2_entries, outfile, indent=4)