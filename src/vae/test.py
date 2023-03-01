import torch, json, os
import pandas as pd
from torch import optim
from data.datareader import *
from data.dataset import MorphDataset
from data.vocab import CharVocab, TagsVocab, TagVocab
from vae.model import VAE
from vae.util import decode_batch

TRN_ID = 1
path = 'results/vae/'+str(TRN_ID)
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
    devset = MorphDataset("data/tur.dev", charvocab=charvocab, tagsvocab=tagsvocab)
    tstbatches = prepare_batches_with_no_pad(devset,1)
    model = VAE()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()
    return model, tstbatches, charvocab

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


def sample(model,  charvocab):
    out_df = pd.DataFrame({})
    N=100
    prior = torch.distributions.normal.Normal(torch.zeros(32), torch.ones(32))
    z = prior.sample((100,)).unsqueeze(0).to(device)
    for i in range(N):
        predtokens = model.sample(z[:,i,:].unsqueeze(0))
        pred_decoded_batches = decode_batch(predtokens, charvocab)[0]
        out_df.at[i, "pred"] = pred_decoded_batches
    out_df.to_csv(path+'/samples.csv')
    #interpolate
    out_df = pd.DataFrame({})
    z1 = z[:,1,:].unsqueeze(0)
    z2 = z[:,5,:].unsqueeze(0)
    n= 10
    for i in range(n):
        cof = 1.0*i/(n-1)
        print(cof)
        zi = torch.lerp(z1, z2, cof)
        predtokens = model.sample(zi)
        pred_decoded_batches = decode_batch(predtokens, charvocab)[0]
        out_df.at[i, "sample"] = pred_decoded_batches
    out_df.to_csv(path+'/interpolations.csv')



def interpolate(model,  charvocab, word1, word2):
    w1 = torch.tensor(charvocab.encode(word1)).unsqueeze(0).to(device)
    z1,_,_, _ = model.encoder(w1)
    w2 = torch.tensor(charvocab.encode(word2)).unsqueeze(0).to(device)
    z2,_,_, _ = model.encoder(w2)
    out_df = pd.DataFrame({})
    out_df.at[0, "sample"] = word1
    n= 10
    for i in range(1,n):
        cof = 1.0*i/(n-1)
        print(cof)
        zi = torch.lerp(z1, z2, cof)
        predtokens = model.sample(zi)
        pred_decoded_batches = decode_batch(predtokens, charvocab)[0]
        out_df.at[i, "sample"] = pred_decoded_batches
    out_df.at[i+1, "sample"] = word2

    out_df.to_csv(path+'/interpolation_between2words.csv')


if __name__ == "__main__":
    MODEL_PATH = path+"/vae_epc48.pt"
    CHARVOCAB_PATH = path+"/charvocab.json"
    model, tst, charvocab = config(MODEL_PATH, CHARVOCAB_PATH)
    #test(model)
    sample(model,charvocab)
    interpolate(model,charvocab, "geliyorum", "okudular")
