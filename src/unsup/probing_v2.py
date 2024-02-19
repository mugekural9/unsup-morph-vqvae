import torch, os, logging, shutil, random, json, sys
import pandas as pd
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from dataset.morph_dataset import MorphDataset
from dataset.datareader import *
from unsup.model_v2 import VQVAE
from unsup.probe import Probe
from util import accuracy_on_batch, decode_batch
import numpy as np
import h5py

def my_custom_logger(logger_name, level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
                    "— %(message)s")
    log_format = logging.Formatter(format_string)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(path+"/probing/"+PROBE_KEY+"/"+PROBING_TYPE+"_probing_vqvae.log_"+logger_name, mode='w')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    return logger


device = "cuda" if torch.cuda.is_available() else "cpu"
DICT_NAME = "6x8"
EPC_NAME  = "429"
RUN_NAME  = "8"
KL_NAME   = "0.05"
PersonMERGED = True

TRN_ID = "run"+RUN_NAME+"_"+DICT_NAME+"_zLEM100_zTAG128_decnh256_kl"+KL_NAME+"_epc10_strt10_decdo0.1_inpemb128_bsize16"
path = 'results/vqvae/unsup/sharedtask/'+str(TRN_ID)

BATCHSIZE = 512
EPOCHS = 10
LR = 0.0005
ENC_NH = 256
DEC_NH = 256
Z_LEMMA_NH = 100
Z_TAG_NH = 128
NUM_CODEBOOK = 6
NUM_CODEBOOK_ENTRIES = 8
BIDIRECTIONAL = True
DEC_DROPOUT = 0.0
INPUT_EMB_DIM = 128


'''{'Part': <dataset.vocab.TagVocab object at 0x7f1226431f60>, 
 'PersonMERGED': <dataset.vocab.TagVocab object at 0x7f1226431fc0>, 
 'Polarity': <dataset.vocab.TagVocab object at 0x7f1226432020>, 
 'Tense': <dataset.vocab.TagVocab object at 0x7f1226432080>,
'Valency': <dataset.vocab.TagVocab object at 0x7f12264320e0>, 
'Language-Specific': <dataset.vocab.TagVocab object at 0x7f1226432140>,
'Aspect': <dataset.vocab.TagVocab object at 0x7f12264321a0>, 
'Interrogativity': <dataset.vocab.TagVocab object at 0x7f1226432200>, 
'Mood': <dataset.vocab.TagVocab object at 0x7f1226432260>, 
'Finiteness': <dataset.vocab.TagVocab object at 0x7f12264322c0>}'''

'''{'Part': <dataset.vocab.TagVocab object at 0x7fdb9dc35d20>, 'Argument': <dataset.vocab.TagVocab object at 0x7fdb9dc35d80>, 'Number': <dataset.vocab.TagVocab object at 0x7fdb9dc35de0>, 'Tense': <dataset.vocab.TagVocab object at 0x7fdb9dc35e40>, 'PersonMERGED': <dataset.vocab.TagVocab object at 0x7fdb9dc35ea0>, 'Polarity': <dataset.vocab.TagVocab object at 0x7fdb9dc35f00>, 'Valency': <dataset.vocab.TagVocab object at 0x7fdb9dc35f60>, 'Language-Specific': <dataset.vocab.TagVocab object at 0x7fdb9dc35fc0>, 'Interrogativity': <dataset.vocab.TagVocab object at 0x7fdb9dc36020>, 'Case': <dataset.vocab.TagVocab object at 0x7fdb9dc36080>, 'Possession': <dataset.vocab.TagVocab object at 0x7fdb9dc360e0>, 'Aspect': <dataset.vocab.TagVocab object at 0x7fdb9dc36140>, 'Mood': <dataset.vocab.TagVocab object at 0x7fdb9dc361a0>, 'Finiteness': <dataset.vocab.TagVocab object at 0x7fdb9dc36200>}'''

if PersonMERGED:
   
    tag_ids = {
        "Part" : 0,
        "PersonMERGED": 1,
        "Polarity": 2,
        "Tense": 3,
        "Valency": 4,
        "Language-Specific": 5,
        "Aspect": 6,
        "Interrogativity": 7,
        "Mood" : 8,
        "Finiteness": 9}
    
    ''' tag_ids = {
        "Part" : 0,
        "Argument": 1,
        "Number": 2,
        "Tense": 3,
        "PersonMERGED": 4,
        "Polarity": 5,
        "Valency": 6,
        "Language-Specific": 7,
        "Interrogativity" : 8,
        "Case": 9,
        "Possession":10,
        "Aspect" : 11,
        "Mood": 12,
        "Finiteness":13
        }'''
     
    """tag_ids = {
        "Part" : 0,
        "Mood" : 1,
        "Tense": 2,
        "Aspect": 3,
        "PersonMERGED": 4,
        "Number": 5,
        "Valency": 6,
        "Polarity": 7,
        "Interrogativity": 8,
        "Language-Specific": 9,
        "Finiteness": 10}"""
else:
    tag_ids = {
    "Part" : 0,
    "Mood" : 1,
    "Tense": 2,
    "Aspect": 3,
    "Person": 4,
    "Number": 5,
    "Valency": 6,
    "Polarity": 7,
    "Interrogativity": 8,
    "Language-Specific": 9,
    "Finiteness": 10}

"""{'Part': <dataset.vocab.TagVocab object at 0x7f2ee55ca2f0>, 
 'Mood': <dataset.vocab.TagVocab object at 0x7f2ee55ca350>, 
 'Tense': <dataset.vocab.TagVocab object at 0x7f2ee55ca3b0>, 
 'Aspect': <dataset.vocab.TagVocab object at 0x7f2ee55ca410>, 
 #'Person': <dataset.vocab.TagVocab object at 0x7f2ee55ca470>, 
 'PersonMERGED': <dataset.vocab.TagVocab object at 0x7fc2fe325ba0>, 
 'Number': <dataset.vocab.TagVocab object at 0x7f2ee55ca4d0>, 
 'Valency': <dataset.vocab.TagVocab object at 0x7f2ee55ca530>, 
 'Polarity': <dataset.vocab.TagVocab object at 0x7f2ee55ca590>, 
 'Interrogativity': <dataset.vocab.TagVocab object at 0x7f2ee55ca5f0>, 
 'Language-Specific': <dataset.vocab.TagVocab object at 0x7f2ee55ca650>, 
 'Finiteness': <dataset.vocab.TagVocab object at 0x7f2ee55ca6b0>}"""





def config(PROBE_KEY, PROBING_TYPE, CODEBOOK_ID, TAG_ID, logger):

    if PROBING_TYPE == "per_codebook":
        PROBE_SIZE = 128
        PROBING_TYPE = "per_codebook" + str(CODEBOOK_ID)
    if PROBING_TYPE == "sum_codebook":
        PROBE_SIZE = 128
    if PROBING_TYPE == "lemma":
        PROBE_SIZE = 100

    logger.info("RUN_NAME:  %s" % RUN_NAME)
    logger.info("DICT_NAME: %s" % DICT_NAME)
    logger.info("EPC_NAME:  %s" % EPC_NAME)
    logger.info("KL_NAME:   %s" % KL_NAME)

    logger.info("PROBE_KEY: %s" % PROBE_KEY)
    logger.info("TAG_ID: %d" % TAG_ID)
    if CODEBOOK_ID:
        logger.info("DICT_ID: %d"% CODEBOOK_ID)

    #logger.basicConfig(handlers=[
    #        logger.FileHandler(path+"/probing/"+PROBE_KEY+"/"+PROBING_TYPE+"_probing_vqvae.log"),
    #        logger.StreamHandler()],
    #        format='%(asctime)s - %(message)s', level=logger.INFO)
    
    if PersonMERGED:
        #trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur/all_verbs_copy.merged")
        trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur/tur_large_copy.train_verbs")
        #trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur/tur_large_copy.train")

    else:
        trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur/all_verbs.merged")
    tstset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS_"+PROBE_KEY+"_filter", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)

    tstbatches = prepare_batches_with_no_pad_wlemmas(tstset, batchsize=1)
    dictmeta = []
    for tagkey,tagvalues in trainset.tagsvocab.vocabs.items():
        dictmeta.append(len(tagvalues.id2tag))

    vocabsize = len(trainset.charvocab.char2id)
    model = VQVAE(vocabsize, dictmeta, ENC_NH, DEC_NH, Z_LEMMA_NH, Z_TAG_NH, DEC_DROPOUT, INPUT_EMB_DIM, NUM_CODEBOOK, NUM_CODEBOOK_ENTRIES, BIDIRECTIONAL)
    model.load_state_dict(torch.load("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/results/vqvae/unsup/sharedtask/run"+RUN_NAME+"_"+DICT_NAME+"_zLEM100_zTAG128_decnh256_kl"+KL_NAME+"_epc10_strt10_decdo0.1_inpemb128_bsize16/vqvae_"+EPC_NAME+".pt"))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    model.to(device)
    probe = Probe(PROBE_SIZE, len(trainset.tagsvocab.vocabs[PROBE_KEY].id2tag))
    logger.info(trainset.tagsvocab.vocabs)
    probe.to(device)
    opt = optim.Adam(probe.parameters(), lr=LR,  betas=(0.5, 0.999))
    return model, opt, tstbatches, trainset.charvocab, trainset.tagsvocab, probe

def train(model, opt, tstbatches, charvocab, tagsvocab, probe, PROBE_KEY, PROBING_TYPE, CODEBOOK_ID, TAG_ID, logger):
    tstsize = sum([len(i[0]) for i in tstbatches.values()])
    logger.info("tstsize: %d" % tstsize)
    accs = []
    for epc in range(EPOCHS):
        two_logits = []
        three_logits = []
        four_logits = []
        out_df = pd.DataFrame({})
        logger.info("")
        logger.info("#epc: %d" % epc)
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
            tgt,tags,lemma = batch
            opt.zero_grad()
            _, _, _, _, _, _, (lemma, tmp_quantized_z),_ = model(tgt)

            if PROBING_TYPE == "lemma":
                #lemma
                loss, pred_token,true,total, logits = probe(lemma.detach(), tags[:,TAG_ID])
            if "per_codebook" in PROBING_TYPE:
                #one-dict
                z = tmp_quantized_z[CODEBOOK_ID].permute(1,0,2)
                loss, pred_token,true,total, logits = probe(z.detach(), tags[:,TAG_ID])
           
            if PROBING_TYPE == "sum_codebook":
                #sum over all codebooks
                #(1,batchsize,dec_nh)
                z = torch.sum(torch.stack(tmp_quantized_z),dim=0)
                loss, pred_token,true,total, logits = probe(z.detach(), tags[:,TAG_ID])
            
            if pred_token.item()==2:
                two_logits.append( logits[0,0,2:].tolist())
            if pred_token.item()==3:
                three_logits.append( logits[0,0,2:].tolist())
            if pred_token.item()==4:
                four_logits.append( logits[0,0,2:].tolist())

            #dataset statistics
            for value in tagsvocab.vocabs[PROBE_KEY].tag2id.values():
                tdict[value]   += tags[:,TAG_ID].tolist().count(value)
            loss.backward()
            opt.step()
            epc_loss += loss.item()
            epc_true+=true
            epc_total+=total
        epc_acc = epc_true/epc_total
        accs.append(epc_acc)
        logger.info("epc: %d, epc_loss: %.3f, epc_acc: %.3f" % (epc, epc_loss, epc_acc))
        if PROBING_TYPE == "per_codebook" and CODEBOOK_ID == 1:
            with h5py.File(str(epc)+'_2.h5', 'w') as f:
                dset = f.create_dataset("default", data=two_logits)
            with h5py.File(str(epc)+'_3.h5', 'w') as f:
                dset = f.create_dataset("default", data=three_logits)
            with h5py.File(str(epc)+'_4.h5', 'w') as f:
                dset = f.create_dataset("default", data=four_logits)
    logger.info(tdict)
    logger.info("baseline: %.4f" %  (max(tdict.values())/tstsize))
    logger.info("max acc: %.4f" % max(accs))
    #out_df.to_csv('2x64_probe_results.csv')


def main(PROBING_TYPE, PROBE_KEY, TAG_ID):
    if PROBING_TYPE == "per_codebook":
        for CODEBOOK_ID in range(NUM_CODEBOOK):
            logger = my_custom_logger(f"Logger{CODEBOOK_ID}_{PROBE_KEY}")
            model, opt, tst,  charvocab, tagsvocab, probe = config(PROBE_KEY, PROBING_TYPE, CODEBOOK_ID,TAG_ID, logger)
            train(model, opt, tst, charvocab, tagsvocab, probe, PROBE_KEY, PROBING_TYPE, CODEBOOK_ID, TAG_ID, logger)
            print("----------------")
    elif PROBING_TYPE == "sum_codebook":
        logger = my_custom_logger(f"Logger_sum_codebook_{PROBE_KEY}")
        model, opt, tst,  charvocab, tagsvocab, probe = config(PROBE_KEY, PROBING_TYPE, None,TAG_ID, logger)
        train(model, opt, tst, charvocab, tagsvocab, probe, PROBE_KEY, PROBING_TYPE, None, TAG_ID, logger)
    elif PROBING_TYPE == "lemma":
        logger = my_custom_logger(f"Logger_lemma_{PROBE_KEY}")
        model, opt, tst,  charvocab, tagsvocab, probe = config(PROBE_KEY, PROBING_TYPE, None, TAG_ID, logger)
        train(model, opt, tst, charvocab, tagsvocab, probe, PROBE_KEY, PROBING_TYPE, None, TAG_ID, logger)



if __name__ == '__main__':
    
    if PersonMERGED:
        probe_keys = ["PersonMERGED","Tense","Polarity"]
    else:
        probe_keys = ["Person", "Tense", "Polarity"]

    for PROBE_KEY in probe_keys:
        TAG_ID = tag_ids[PROBE_KEY]
        #for PROBING_TYPE in ["lemma", "per_codebook", "sum_codebook"]:
        for PROBING_TYPE in ["per_codebook",]:
            main(PROBING_TYPE, PROBE_KEY, TAG_ID)


