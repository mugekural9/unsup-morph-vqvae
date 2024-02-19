added_nouns = []
added_verbs = []
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train", "r") as reader:
    for line in reader:
        lemma, tgt, tags = line.split('\t')
        if tags.split(';')[0] == 'N':
            added_nouns.append(line)
        elif tags.split(';')[0] == 'V':
            added_verbs.append(line)


with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train_nouns", "w") as writer:
    for line in added_nouns:
        writer.write(line)


with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train_verbs", "w") as writer:
    for line in added_verbs:
        writer.write(line)