import torch, json, os
import pandas as pd
from torch import optim
from data.datareader import *
from data.dataset import MorphDataset
from data.vocab import CharVocab, TagsVocab, TagVocab
from ae.model import AE
from ae.util import decode_batch

TRN_ID = 1
path = 'results/ae/'+str(TRN_ID)
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
    devset = MorphDataset("data/tur.dev", charvocab=charvocab, tagsvocab=tagsvocab)
    trnbatches = prepare_batches_with_no_pad(trainset,1)
    tstbatches = prepare_batches_with_no_pad(devset,1)
    model = AE()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    return model, tstbatches, charvocab, trnbatches

def test(model, tstbatches, charvocab):
    tstsize = sum([len(i) for i in tstbatches.values()])
    out_df = pd.DataFrame({})
    model.eval()
    total_exact_match = 0
    i=0
    for bid, batch in tstbatches.items():
        predtokens = model.decode(batch)
        gold_decoded_batches = decode_batch(batch[:,1:], charvocab)
        pred_decoded_batches = decode_batch(predtokens, charvocab)

        for g,p in zip(gold_decoded_batches, pred_decoded_batches):
            i+=1
            out_df.at[i, "gold"] = g
            out_df.at[i, "pred"] = p
            if g == p:
                out_df.at[i, "exact_match"] = 1
            else:
                out_df.at[i, "exact_match"] = 0
    total_exact_match =  sum(out_df.exact_match==1)
    print("exact_match: %.3f" % (total_exact_match/tstsize))
    out_df.to_csv(path+'/preds.csv')



def save_hiddens(model, tstbatches, charvocab):
    tstsize = sum([len(i) for i in tstbatches.values()])
    out_df = pd.DataFrame({})
    model.eval()
    total_exact_match = 0
    i=0
    zdict = dict()
    for bid, batch in tstbatches.items():
        predtokens, z = model.decode(batch)
        zdict[i] = z
        gold_decoded_batches = decode_batch(batch[:,1:], charvocab)
        pred_decoded_batches = decode_batch(predtokens, charvocab)

        for g,p in zip(gold_decoded_batches, pred_decoded_batches):
            i+=1
            out_df.at[i, "gold"] = g
            out_df.at[i, "pred"] = p
            if g == p:
                out_df.at[i, "exact_match"] = 1
            else:
                out_df.at[i, "exact_match"] = 0
        if i==500:
            break
    total_exact_match =  sum(out_df.exact_match==1)
    print("exact_match: %.3f" % (total_exact_match/tstsize))
    torch.save(zdict, path+'/500trnwords_z.pt')

if __name__ == "__main__":
    MODEL_PATH = path+"/ae.pt"
    CHARVOCAB_PATH = path+"/charvocab.json"
    model, tst, charvocab, trn = config(MODEL_PATH, CHARVOCAB_PATH)
    #test(model)
    with torch.no_grad():
        save_hiddens(model, trn, charvocab)