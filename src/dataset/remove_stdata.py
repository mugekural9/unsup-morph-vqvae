words = dict()
with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        words[word] = line

sharedtask_words = dict()
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        sharedtask_words[word] = line

with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.dev") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        sharedtask_words[word] = line

with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        sharedtask_words[word] = line


with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_removed_stdata", "w") as writer:
    for word, line in words.items():
        if word not in sharedtask_words:
            writer.write(line)

