import json

class CharVocab():
    def __init__(self, words=None):
        self.id2char = {0:'pad', 1:'<unk>', 2: '<s>', 3:'</s>'}
        self.char2id = {'pad':0, '<unk>':1, '<s>':2, '</s>':3}
        if words is not None:
            for word in words:
                for char in word:
                    if char not in self.char2id:
                        self.char2id[char] = len(self.char2id)
                        self.id2char[ self.char2id[char]] = char
    
    def load_from_dict(self, dict):
        for id, char in dict.items():
            id = int(id)
            self.id2char[id] = char
            self.char2id[char] = id

    def get_char_id(self, char):
        if char not in self.char2id:
            return self.char2id['<unk>']
        else:
            return self.char2id[char]

    def get_id_char(self, id):
        if id not in self.id2char:
            return self.id2char['<unk>']
        else:
            return self.id2char[id]

    def encode(self, word):
        return [self.char2id['<s>']] + [self.get_char_id(w) for w in word] + [self.char2id['</s>']]

class TagsVocab():
    with open('data/unischema_tags.json') as json_file:
        unimorphtags = json.load(json_file)
    def __init__(self, tagslist=None):
        self.vocabs = {}
        if tagslist is not None:
            for tags in tagslist:
                for tag in tags.split(';'):
                    tag = tag.lower()
                    tagtype = self.unimorphtags[tag]
                    if tagtype not in self.vocabs:
                        self.vocabs[tagtype] = TagVocab()
                    self.vocabs[tagtype].add(tag)


    def encode(self, tags):
        ids = []
        for tag in tags.split(';'):
            tag = tag.lower()
            tagtype = self.unimorphtags[tag]
            ids.append(self.vocabs[tagtype].encode(tag))
        return ids

class TagVocab():
    def __init__(self):
       self.id2tag = {0:'pad', 1:'<unk>'}
       self.tag2id = {'pad':0, '<unk>':1}
    
    def load_from_dict(self, dict):
        for id, tag in dict.items():
            id = int(id)
            self.id2tag[id] = tag
            self.tag2id[tag] = id

    def add(self, tag):
        if tag not in self.tag2id:
            self.tag2id[tag] = len(self.tag2id)
            self.id2tag[self.tag2id[tag]] = tag

    def encode(self, tag):
        return self.tag2id[tag]
