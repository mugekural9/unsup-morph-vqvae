import torch, json, os
import pandas as pd
from torch import optim
from data.datareader import *
from data.dataset import MorphDataset
from data.vocab import CharVocab, TagsVocab, TagVocab
from vqvae.model import VQVAE
from vqvae.util import decode_batch

TRN_ID = 1
path = 'results/vqvae/'+str(TRN_ID)
device = "cuda" if torch.cuda.is_available() else "cpu"

def config(MODEL_PATH, CHARVOCAB_PATH):
    charvocab = CharVocab()
    with open(CHARVOCAB_PATH, 'r') as openfile:
        chardict = json.load(openfile)
        charvocab.load_from_dict(chardict)
    tagsvocab = TagsVocab()
    files= os.listdir(path)
    for fs in files:
        if "_vocab.json" in fs:
            tag = fs.split("_vocab.json")[0]
            tagvocab = TagVocab()
            with open(path+"/"+fs, 'r') as openfile:
                tagdict= json.load(openfile)
                tagvocab.load_from_dict(tagdict)
                tagsvocab.vocabs[tag] = tagvocab
    trainset = MorphDataset("data/tur_large.train")
    maxtrnsize = 10000
    trainset.lemmas = trainset.lemmas[:maxtrnsize]
    trainset.tgts = trainset.tgts[:maxtrnsize]
    trainset.tagslist = trainset.tagslist[:maxtrnsize]

    trnbatches, trnbatches_lemma = prepare_batches_with_no_pad_with_lemmas(trainset,1)
    model = VQVAE()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    return model, charvocab, trnbatches, trnbatches_lemma

def test(model, batches, charvocab, batches_lemma):
    out_df = pd.DataFrame({})
    model.eval()
    i=0
    for bid, batch in batches.items():
        batch_lemma = batches_lemma[bid]
        _, _, _, _,  quantized_indices = model(batch)
        predtokens = model.decode(batch)
        gold_decoded_batches = decode_batch(batch[:,1:], charvocab)
        pred_decoded_batches = decode_batch(predtokens, charvocab)
        lemma_batches = decode_batch(batch_lemma[:,1:], charvocab)

        for g,p,lemma in zip(gold_decoded_batches, pred_decoded_batches, lemma_batches):
            out_df.at[i, "lemma"] = lemma
            out_df.at[i, "gold"] = g
            out_df.at[i, "pred"] = p

            out_df.at[i, "codebook_entry"] = quantized_indices.item()
            if g == p:
                out_df.at[i, "exact_match"] = 1
            else:
                out_df.at[i, "exact_match"] = 0
            i+=1
    out_df.to_csv(path+'/preds.csv')



if __name__ == "__main__":
    MODEL_PATH = path+"/vqvae.pt"
    CHARVOCAB_PATH = path+"/charvocab.json"
    model, charvocab, trn, lemmas = config(MODEL_PATH, CHARVOCAB_PATH)
    with torch.no_grad():
        test(model, trn, charvocab, lemmas)
        