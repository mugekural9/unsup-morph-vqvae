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
BATCHSIZE = 64
KL_WEIGHT = 0.1
KL_ANNEAL_EPC = 10
KL_START_EPC = 5
ENC_NH = 256
DEC_NH = 350
Z_NH = 128
DEC_DROPOUT = 0.2
INPUT_EMB_DIM = 64
TRN_ID = "tur_mergedstdata_VERBS_256k_z"+str(Z_NH)+"_dec_nh"+str(DEC_NH)+"_kl"+str(KL_WEIGHT)+"_epc"+str(KL_ANNEAL_EPC)+"_start"+str(KL_START_EPC)+"_dec_dropout"+str(DEC_DROPOUT)+"_input_emb_dim"+str(INPUT_EMB_DIM)+"_batchsize"+str(BATCHSIZE)
path = 'results/vqvae/'+str(TRN_ID)
    
def config():

    trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata_VERBS")
    devset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.dev_VERBS", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
    tstset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)

    maxtrnsize = 256000
    trainset.lemmas = trainset.lemmas[:maxtrnsize]
    trainset.tgts = trainset.tgts[:maxtrnsize]
    trainset.tagslist = trainset.tagslist[:maxtrnsize]
  
    maxdevsize = maxtrnsize
    devset.lemmas = devset.lemmas[:maxdevsize]
    devset.tgts = devset.tgts[:maxdevsize]
    devset.tagslist = devset.tagslist[:maxdevsize]
      
  
    tstbatches = prepare_batches_with_no_pad_wlemmas(tstset, batchsize=1)
    dictmeta = []
    for tagkey,tagvalues in trainset.tagsvocab.vocabs.items():
        print(tagkey,"->", len(tagvalues.id2tag))
        dictmeta.append(len(tagvalues.id2tag))
    vocabsize = len(trainset.charvocab.char2id)
    model = VQVAE(vocabsize,dictmeta, ENC_NH, DEC_NH, Z_NH, DEC_DROPOUT, INPUT_EMB_DIM)
    model.load_state_dict(torch.load("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/results/vqvae/tur_mergedstdata_VERBS_256k_z128_dec_nh350_kl0.1_epc10_start5_dec_dropout0.2_input_emb_dim64_batchsize64/vqvae_36.pt"))
    model.to(device)
    model.eval()
    return model, tstbatches, trainset.charvocab

def reinflect(model, tstbatches, charvocab):
    tgt,tags,lemma = tstbatches[1]
    tgt =  torch.tensor(charvocab.encode("geldiler")).unsqueeze(0).to(device)

    out_df = pd.DataFrame({})
    part        = 0
    number      = 4
    person      = 3
    polarity    = 2
    tense       = 3
    valency     = 2
    lspec       = 0
    aspect      = 3
    inter       = 0
    mood        = 2
    finite      = 0
    
    verb_entries =  [2,3,4,2,4,2,2,3,0,2,0]
    #all[3, 4, 5, 4, 5, 3, 5, 8, 3, 5, 3]
 
    unique_words = defaultdict(lambda:0)
    i = 0
    for part in range(3):
        for number in range(4):
            for person in range(5):
                for polarity in range(4):
                    for tense in range(5):
                        for valency in range(3):
                            entries  = [part, number,person,2,3, valency,  4,7,2,4,2]
                            predtokens = model.reinflect(tgt, entries)
                            decoded_batches = decode_batch(predtokens, charvocab)[0]
                            unique_words[decoded_batches] +=1
                            i+=1
                            out_df.at[i, "generated"] = decoded_batches
    out_df.to_csv('reinflections.csv')
    print("#unique  words: %d" %  len(unique_words))

model, tstbatches, charvocab = config()
reinflect(model, tstbatches, charvocab)
