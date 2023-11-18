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
DEC_NH = 512
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
        #print(tagkey,"->", len(tagvalues.id2tag))
        dictmeta.append(len(tagvalues.id2tag))
    vocabsize = len(trainset.charvocab.char2id)
    model = VQVAE(vocabsize,dictmeta, ENC_NH, DEC_NH, Z_NH, DEC_DROPOUT, INPUT_EMB_DIM)
    model.load_state_dict(torch.load("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/results/vqvae/tur_mergedstdata_VERBS_256k_z128_dec_nh512_kl0.1_epc10_start5_dec_dropout0.2_input_emb_dim64_batchsize64/vqvae_40.pt"))
    model.to(device)
    model.eval()
    return model, tstbatches, trainset.charvocab

def reinflect(model, tstbatches, charvocab):
    ## tst (decoding for for copying dict-supervised with lemma)
    out_df = pd.DataFrame({})
    i=0
    exact_match_tst_dictsupervised_wlemma = 0
    for bid, batch in tstbatches.items():
        tgt,tags,lemma = batch
        predtokens = model.decode(lemma,tags)
        lemmas = decode_batch(lemma[:,1:], charvocab)
        gold_decoded_batches = decode_batch(tgt[:,1:], charvocab)
        pred_decoded_batches = decode_batch(predtokens, charvocab)
        for g,p,l in zip(gold_decoded_batches, pred_decoded_batches, lemmas):
            out_df.at[i, "gold"] = g
            out_df.at[i, "pred"] = p
            out_df.at[i, "lemma"] = l
            centry = ""
            for k in range(len(tags[0])):
                centry += str(tags[0][k].item()) + "-"
            out_df.at[i, "FULL_codebook_entry"] = centry
            if g == p:
                out_df.at[i, "exact_match"] = 1
                exact_match_tst_dictsupervised_wlemma+=1
            else:
                out_df.at[i, "exact_match"] = 0
            i+=1
    print("exact_match_tst_dictsupervised_wlemma:", exact_match_tst_dictsupervised_wlemma/i)
    out_df.to_csv('sup_wlemmas.csv')

model, tstbatches, charvocab = config()
reinflect(model, tstbatches, charvocab)
