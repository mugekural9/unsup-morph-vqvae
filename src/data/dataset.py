from data.vocab import CharVocab, TagsVocab

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
                self.lemmas.append(lemma)
                self.tgts.append(tgt)
                self.tagslist.append(tags)
                if False: #'train' in datafile: #workaround to include lemmas in training as well
                    self.lemmas.append(lemma)
                    self.tgts.append(lemma)
                    self.tagslist.append(tags)
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
        flat_list = [item for sublist in tgt_ids for item in sublist]
        num_unks = flat_list.count(1)
        print("num_unks: %d" % num_unks)
        return lemma_ids, tgt_ids, tags_ids           
