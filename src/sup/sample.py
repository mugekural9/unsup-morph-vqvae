import torch, os, logging, shutil, random, json
import pandas as pd
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from dataset.morph_dataset import MorphDataset
from dataset.datareader import *
from ae.model import AE
from sup.model import VQVAE
from util import accuracy_on_batch, decode_batch

device = "cuda" if torch.cuda.is_available() else "cpu"
TRN_ID = "311_tur_mergedstdata_128k_z64_generations"
path = 'results/vqvae/'+str(TRN_ID)
    
def config():
    trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata")
    dictmeta = []
    for tagkey,tagvalues in trainset.tagsvocab.vocabs.items():
        dictmeta.append(len(tagvalues.id2tag))
    vocabsize = len(trainset.charvocab.char2id)
    model = VQVAE(vocabsize,dictmeta)
    model.load_state_dict(torch.load("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/results/vqvae/311_tur_mergedstdata_128k_z32/vqvae.pt"))
    model.to(device)
    return model,  trainset.charvocab

def sample(model, charvocab):
    out_df = pd.DataFrame({})
    for i in range(50):
        predtokens = model.sample()
        decoded_batches = decode_batch(predtokens, charvocab)
        print(decoded_batches)
        #out_df.at[i, "generated"] = decoded_batches
        #out_df.to_csv(path+'/generations.csv')

model, charvocab = config()
sample(model, charvocab)
