import torch, os, logging, shutil, random, json
import pandas as pd
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from dataset.morph_dataset import MorphDataset
from dataset.datareader import *
from sup.model import VQVAE
from sup.probe import Probe

from util import accuracy_on_batch, decode_batch


device = "cuda" if torch.cuda.is_available() else "cpu"
TRN_ID = "211_turlarge_train"
path = 'results/vqvae/'+str(TRN_ID)
    
#random.seed(0)
BATCHSIZE = 128
EPOCHS = 1000
LR = 0.0005
#{'Part': <dataset.vocab.TagVocab object at 0x7f3dd43cea40>, 'Argument': <dataset.vocab.TagVocab object at 0x7f3dd43ceaa0>, 'Number': <dataset.vocab.TagVocab object at 0x7f3dd43ceb00>, 'Tense': <dataset.vocab.TagVocab object at 0x7f3dd43ceb60>, 'Person': <dataset.vocab.TagVocab object at 0x7f3dd43cebc0>, 'Polarity': <dataset.vocab.TagVocab object at 0x7f3dd43cec20>, 'Valency': <dataset.vocab.TagVocab object at 0x7f3dd43cec80>, 'Language-Specific': <dataset.vocab.TagVocab object at 0x7f3dd43cece0>, 'Interrogativity': <dataset.vocab.TagVocab object at 0x7f3dd43ced40>, 'Case': <dataset.vocab.TagVocab object at 0x7f3dd43ceda0>, 'Possession': <dataset.vocab.TagVocab object at 0x7f3dd43cee00>, 'Aspect': <dataset.vocab.TagVocab object at 0x7f3dd43cee60>, 'Mood': <dataset.vocab.TagVocab object at 0x7f3dd43ceec0>, 'Finiteness': <dataset.vocab.TagVocab object at 0x7f3dd43cef20>}
# Part:0 
# Argument:1 
# Number:2
# Tense:3 
# Person:4 
# Polarity:5 
# Valency:6 
# Language-Specific:7 
# Interrogativity:8 
# Case:9
# Possession:10 
# Aspect:11 
# Mood:12
# Finiteness:13 

PROBE_KEY = 'Tense'
TAG_ID = 3
DICT_ID = 3
print("PROBE_KEY: %s" % PROBE_KEY)
print("TAG_ID: %d" % TAG_ID)
print("DICT_ID: %d"% DICT_ID)


def config():
    trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata")
    #trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train")
    devset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.dev", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
    tstset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)


    tstbatches = prepare_batches_with_no_pad(tstset, batchsize=BATCHSIZE)
    _tstbatches = prepare_batches_with_no_pad(tstset, batchsize=1)

    dictmeta = []
    for tagkey,tagvalues in trainset.tagsvocab.vocabs.items():
        dictmeta.append(len(tagvalues.id2tag))
    vocabsize = len(trainset.charvocab.char2id)
    model = VQVAE(vocabsize,dictmeta)
    model.load_state_dict(torch.load("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/results/vqvae/311_tur_mergedstdata_128k_z64/vqvae.pt"))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    Z_NH = model.z_lemma_mu.out_features
    probe = Probe(Z_NH, len(trainset.tagsvocab.vocabs[PROBE_KEY].id2tag))
    probe.to(device)
    opt = optim.Adam(probe.parameters(), lr=LR,  betas=(0.5, 0.999))
    return model, opt, tstbatches, _tstbatches, trainset.charvocab, trainset.tagsvocab, probe



def train(model, opt, tstbatches, _tstbatches, charvocab, tagsvocab, probe):
    tstsize = sum([len(i[0]) for i in tstbatches.values()])
    logging.info("tstsize: %d" % tstsize)
    
    for epc in range(EPOCHS):
        out_df = pd.DataFrame({})
        logging.info("")
        logging.info("#epc: %d" % epc)
        probe.train()
        keys = list(tstbatches.keys())
        random.shuffle(keys)
        epc_loss = 0
        epc_total = 0
        epc_true = 0
        tdict = dict()
        for value in tagsvocab.vocabs[PROBE_KEY].tag2id.values():
            tdict[value] = 0
        for bid in keys:
            batch = tstbatches[bid]
            tgt,tags = batch
            opt.zero_grad()
            _, _, _, _, _, _, (lemma, tmp_quantized_z) = model(tgt, tags)
            z = tmp_quantized_z[DICT_ID].permute(1,0,2)
            #tgt = decode_batch(tgt[:,1:], charvocab)
            loss, pred_token,true,total = probe(lemma.detach(), tags[:,TAG_ID])
            for value in tagsvocab.vocabs[PROBE_KEY].tag2id.values():
                tdict[value]   += tags[:,TAG_ID].tolist().count(value)
            loss.backward()
            opt.step()
            epc_loss += loss.item()
            epc_true+=true
            epc_total+=total
        epc_acc = epc_true/epc_total
        print("epc: %d, epc_loss: %.3f, epc_acc: %.3f" % (epc, epc_loss, epc_acc))
    print(tdict)
    out_df.to_csv('probe_results.csv')
model, opt, tst, _tst, charvocab, tagsvocab, probe = config()
train(model, opt, tst,_tst, charvocab, tagsvocab, probe)
