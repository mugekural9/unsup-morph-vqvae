import json

f = open('/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/results/vqvae/unsup/verbs/run3_4x8_zLEM100_zTAG128_decnh1024_kl1.0_epc10_strt5_decdo0.2_inpemb128_bsize64/qinds_tst_epc15.json')

data = json.load(f)
 
# Iterating through the json
# list

selections = dict()
for i in range(8):
    selections[i] = []
for key in data.keys():
    i = key.split(";")[3]
    for word in data[key]:
        selections[int(i)].append(word)


with open('cbook3_selections.json', 'w') as fp:
    json.dump(selections, fp, ensure_ascii=False, indent = 4)