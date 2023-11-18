from dataset.vocab import CharVocab, TagsVocab

class MorphDataset():
    def __init__(self, datafile, charvocab=None, tagsvocab=None):
        self.lemmas = []
        self.tgts = []
        self.tagslist = []
        print(datafile)
        ONLY_UNIQUE_DATA = False
        with open(datafile) as reader:
            for line in reader:
                line = line.strip()
                lemma, tgt, tags = line.split('\t')
                if ('train' in datafile) and ONLY_UNIQUE_DATA:
                    #if lemma not in self.tgts:
                    #    self.tgts.append(lemma)
                    #    #workaround to make lists size equal
                    #    self.lemmas.append(lemma)
                    #    self.tagslist.append(tags)
                    if tgt not in self.tgts:
                        self.tgts.append(tgt)
                        #workaround to make lists size equal
                        self.lemmas.append(lemma)
                        self.tagslist.append(tags)
                else:
                    self.lemmas.append(lemma)
                    self.tagslist.append(tags)
                    self.tgts.append(tgt)
    
            if charvocab is None:
                self.charvocab = CharVocab(self.tgts)
            else:
                self.charvocab = charvocab
           
            if tagsvocab is None:
                self.tagsvocab = TagsVocab(self.tagslist)
            else:
                self.tagsvocab = tagsvocab

    def get_tokens(self):
        lemma_ids = []
        tgt_ids = []
        tags_ids = []
        for lemma, tgt, tags in zip(self.lemmas, self.tgts, self.tagslist):
            lemma_ids.append(self.charvocab.encode(lemma))
            tgt_ids.append(self.charvocab.encode(tgt))
            tags_ids.append(self.tagsvocab.encode(tags))
        num_unks_tokens = [item for sublist in tgt_ids for item in sublist].count(1)
        num_unks_tags = [item for sublist in tags_ids for item in sublist].count(1)
        print("num_unks_tokens: %d" % num_unks_tokens)
        print("num_unks_tags: %d" % num_unks_tags)
        return lemma_ids, tgt_ids, tags_ids           
