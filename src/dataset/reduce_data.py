from collections import defaultdict
words = dict()
unique_tags = defaultdict(lambda:0)
twords_dict = dict()
with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata_VERBS") as reader:
    for line in reader:
        lemma, word, _tags = line.split('\t')
        tags = _tags.strip().split(';')
        tags.sort()
        tags = '-'.join(tags)
        unique_tags[tags] += 1
        if tags not in twords_dict:
            twords_dict[tags] = []
        twords_dict[tags].append((lemma,word,_tags))

with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata_VERBS_REDUCED",'w') as writer:
    twords = []
    allwords = []
    alllemmas = dict()
    vcount = 0
    for key,value in  {k: v for k, v in sorted(unique_tags.items(), key=lambda item: item[1])}.items():
        #print(key+"-"+str(value))
        if value>580:
            vcount+=1
            for (lemma,word,tags) in twords_dict[key]:
                allwords.append(word)
                alllemmas[lemma] =1
                ln = lemma + '\t' + word + '\t' + tags 
                writer.write(ln)
    print("num tags: ", vcount)
    print('num words:', len(allwords))
    print('num lemmas:', len(alllemmas))

    #filter that dataset, and train
    #num tags:  356, bu ayni gelmiyorr???
    #num words: 207471
    #num lemmas: 588

