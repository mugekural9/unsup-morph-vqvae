word_tags = dict()
with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates") as reader:
    for line in reader:
        line = line.strip()
        _, word, tags = line.split('\t')
        word_tags[word] = tags.split(';')

sharedtask_word_tags = dict()
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train") as reader:
    for line in reader:
        line = line.strip()
        _, word, tags = line.split('\t')
        sharedtask_word_tags[word] = tags.split(';')

i = 0
eq = 0
neq = 0
neq_words = []
with open("neqs.txt", "w") as writer:
    for word, tags in sharedtask_word_tags.items():
        i+=1
        if(set(tags) == set(word_tags[word])):
            eq += 1
        else:
            neq +=1
            neq_words.append((word,tags))
            writer.write(word+"\t"+str([t for t in tags])+ "\n")

print("i:%d, eq: %d, neq: %d" %(i,eq,neq))
