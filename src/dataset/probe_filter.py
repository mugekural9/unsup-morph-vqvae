import random
lines = []
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS") as reader:
    for line in reader:
        lemma,tgt,tags = line.strip().split("\t")
        if "SG" in tags.split(';') or "PL" in tags.split(';'):
            lines.append(line)

with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS_Person_filter", "w") as writer:
    for line  in lines:
        writer.write(line)
