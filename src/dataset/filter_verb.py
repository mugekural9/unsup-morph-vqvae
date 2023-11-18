words = dict()
with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        tags = tags.split(';')
        if 'V' in tags:
            words[word] = line


with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata_VERBS", "w") as writer:
    for word, line in words.items():
        writer.write(line)

words = dict()
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.dev") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        tags = tags.split(';')
        if 'V' in tags:
            words[word] = line


with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.dev_VERBS", "w") as writer:
    for word, line in words.items():
        writer.write(line)


words = dict()
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        tags = tags.split(';')
        if 'V' in tags:
            words[word] = line


with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS", "w") as writer:
    for word, line in words.items():
        writer.write(line)
