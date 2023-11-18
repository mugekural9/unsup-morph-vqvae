#(unsup-morph) [mugekural@login02 unsup-morph-vqvae]$ py /kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/filter_train.py 
#ALERT IN TRAIN: kapıyor olmalıydınız
#ALERT IN TRAIN: en küçük ortak kat
#ALERT IN TRAIN: en küçük ortak kat
#ALERT IN TRAIN: pelinlerinden
#ALERT IN DEV: porsuğunda
#ALERT IN GOLD: taşıyor olurlar mıydı
#ALERT IN GOLD: kırmızı şarabından
            
sharedtask_word_tags = dict()
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        if word in sharedtask_word_tags:
            print("ALERT IN TRAIN:", word)
        sharedtask_word_tags[word] =line

with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.dev") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        if word in sharedtask_word_tags:
            print("ALERT IN DEV:", word)
        sharedtask_word_tags[word] = line

with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold") as reader:
    for line in reader:
        _, word, tags = line.split('\t')
        if word in sharedtask_word_tags:
            print("ALERT IN GOLD:", word)
        sharedtask_word_tags[word] = line

added_words = dict()
c = 0
with open("/kuacc/users/mugekural/workfolder/dev/tur/tur") as reader:
    with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates", "w") as writer:
        for line in reader:
            _, word, tags = line.split('\t')
            if word not in added_words:
                if word in sharedtask_word_tags:
                    c+=1
                    writer.write(sharedtask_word_tags[word])
                else:    
                    writer.write(line)
            added_words[word] = tags
            
print("%d words from sharedtaskdata" %c)