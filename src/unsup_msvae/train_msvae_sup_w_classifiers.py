import torch, os, logging, shutil, random, json
import pandas as pd
from torch import optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from dataset.morph_dataset import MorphDataset
from dataset.datareader import *
from unsup.model_v2 import VQVAE
from unsup.msvae_sup_w_classifiers import MSVAE
from util import accuracy_on_batch, decode_batch
import math

BATCHSIZE = 128
EPOCHS = 50
LR = 0.0005
KL_ANNEAL_EPC = 10
KL_START_EPC = 5
ENC_NH = 256
DEC_NH = 1024
Z_LEMMA_NH = 100
Z_TAG_NH = 128
BIDIRECTIONAL = True

import argparse
argParser = argparse.ArgumentParser()
argParser.add_argument("--numcodebook", help="numcodebook")
argParser.add_argument("--numentry", help="numentry")
argParser.add_argument("--runid", help="numentry")
argParser.add_argument("--dataset", help="datatype")
argParser.add_argument("--klweight", help="datatype")

args = argParser.parse_args()
NUM_CODEBOOK = int(args.numcodebook)
NUM_CODEBOOK_ENTRIES = int(args.numentry)
RUNID = int(args.runid)
DATASET_TYPE = args.dataset
KL_WEIGHT = float(args.klweight)

DEC_DROPOUT = 0.2
INPUT_EMB_DIM = 128
TRN_ID = 'run'+str(RUNID)+"_"+str(NUM_CODEBOOK)+"x"+str(NUM_CODEBOOK_ENTRIES)+"_zLEM"+str(Z_LEMMA_NH)+"_zTAG"+ str(Z_TAG_NH)+ "_decnh"+str(DEC_NH)+"_kl"+str(KL_WEIGHT)+"_epc"+str(KL_ANNEAL_EPC)+"_strt"+str(KL_START_EPC)+"_decdo"+str(DEC_DROPOUT)+"_inpemb"+str(INPUT_EMB_DIM)+"_bsize"+str(BATCHSIZE)

class uniform_initializer(object):
    def __init__(self, stdv):
        self.stdv = stdv
    def __call__(self, tensor):
        nn.init.uniform_(tensor, -self.stdv, self.stdv)

if DATASET_TYPE == 'all':
    path = 'results/msved/sup_w_classifiers/all/'+str(TRN_ID)
    ffile = '/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold'
elif DATASET_TYPE == 'verbs':
    path = 'results/msved/sup_w_classifiers/verbs/'+str(TRN_ID)
    ffile = '/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS'


try:
    shutil.rmtree(path)
except OSError as error:
    print(error)  

try:
    os.mkdir(path)
except OSError as error:
    print(error)  

#random.seed(0)
writer = SummaryWriter(comment="_MSVED_TRNID_"+str(TRN_ID))
device = "cuda" if torch.cuda.is_available() else "cpu"

tags_to_entries = dict()
word_tags = dict()


def get_temp(update_ind):
    return max(0.5, math.exp(-3 * 1e-5 * update_ind))

def get_kl_weight(update_ind, thres, rate):
    upnum = 1500
    if update_ind <= upnum:
        return 0.0
    else:
        w = (1.0/rate)*(update_ind - upnum)
        if w < thres:
            return w
        else:
            return thres

with open(ffile) as reader:
    for line in reader:
        line = line.strip()
        lemma, tgt, tags = line.split('\t')
        word_tags[tgt] = tags
        tags_to_entries[tags] = defaultdict(lambda:0)

def config():
    logging.basicConfig(handlers=[
            logging.FileHandler(path+"/training_msvae.log"),
            logging.StreamHandler()],
            format='%(asctime)s - %(message)s', level=logging.INFO)
    
    if DATASET_TYPE == 'all':
        trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur/tur_filtered_duplicates_removed_stdata_SHUFFLED_TURLARGE_MERGED")
        devset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
        tstset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
    elif DATASET_TYPE == 'verbs':
        trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur/all_verbs.merged")
        devset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
        tstset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)

    maxtrnsize = 100000
    maxdevsize = 2000
    maxtstsize = 2000

    trainset.lemmas = trainset.lemmas[:maxtrnsize]
    trainset.tgts = trainset.tgts[:maxtrnsize]
    trainset.tagslist = trainset.tagslist[:maxtrnsize]
  
    devset.lemmas = devset.lemmas[:maxdevsize]
    devset.tgts = devset.tgts[:maxdevsize]
    devset.tagslist = devset.tagslist[:maxdevsize]
      
    maxtstsize = maxtstsize
    tstset.lemmas = tstset.lemmas[:maxtstsize]
    tstset.tgts = tstset.tgts[:maxtstsize]
    tstset.tagslist = tstset.tagslist[:maxtstsize]

    logging.info("trnsize: %d" % len(trainset.tgts))
    logging.info("devsize: %d" % len(devset.tgts))
    logging.info("tstsize: %d" % len(tstset.tgts))

    trn_unique_lemmas = defaultdict(lambda:0)
    trn_unique_tgts = defaultdict(lambda:0)
    dev_unique_lemmas = defaultdict(lambda:0)
    dev_unique_tgts = defaultdict(lambda:0)
    tst_unique_lemmas = defaultdict(lambda:0)
    tst_unique_tgts = defaultdict(lambda:0)
    for lemma in trainset.lemmas:
        trn_unique_lemmas[lemma] += 1
    for tgt in trainset.tgts:
        trn_unique_tgts[tgt] +=1
    for lemma in devset.lemmas:
        dev_unique_lemmas[lemma] += 1
    for tgt in devset.tgts:
        dev_unique_tgts[tgt] +=1
    for lemma in tstset.lemmas:
        tst_unique_lemmas[lemma] += 1
    for tgt in tstset.tgts:
        tst_unique_tgts[tgt] +=1
    seen_dev_tgt = 0
    for tgt in devset.tgts:
        if tgt in trn_unique_tgts:
            seen_dev_tgt +=1
    seen_dev_lemma = 0
    for lemma in devset.lemmas:
        if lemma in trn_unique_lemmas:
            seen_dev_lemma +=1
    seen_tst_tgt = 0
    for tgt in tstset.tgts:
        if tgt in trn_unique_tgts:
            seen_tst_tgt +=1
    seen_tst_lemma = 0
    for lemma in tstset.lemmas:
        if lemma in trn_unique_lemmas:
            seen_tst_lemma +=1
    seen_dev_taglist=0
    seen_tst_taglist=0
    trn_tagsets = []
    trn_tagsets_dict = defaultdict(lambda:0)
    dev_tagsets_dict = defaultdict(lambda:0)
    tst_tagsets_dict = defaultdict(lambda:0)

    for taglist in trainset.tagslist:
        _tgs = taglist.split(';')
        _tgs.sort()
        _tgs = '-'.join(_tgs)
        trn_tagsets.append(_tgs)
        trn_tagsets_dict[_tgs]+=1

    for taglist in devset.tagslist:
        _tgs = taglist.split(';')
        _tgs.sort()
        _tgs = '-'.join(_tgs)
        if _tgs in trn_tagsets:
            seen_dev_taglist +=1
        dev_tagsets_dict[_tgs] += 1

    for taglist in tstset.tagslist:
        _tgs = taglist.split(';')
        _tgs.sort()
        _tgs = '-'.join(_tgs)
        if _tgs in trn_tagsets:
            seen_tst_taglist +=1
        tst_tagsets_dict[_tgs] += 1


    logging.info("Trn unique tgts: %d" % len(trn_unique_tgts))
    logging.info("Trn unique lemmas: %d" % len(trn_unique_lemmas))
    logging.info("Trn unique taglists: %d" % len(trn_tagsets_dict))
    
    logging.info("Dev unique tgts: %d" % len(dev_unique_tgts))
    logging.info("Dev unique lemmas: %d" % len(dev_unique_lemmas))
    logging.info("Dev unique taglists: %d" % len(dev_tagsets_dict))
    
    logging.info("Tst unique tgts: %d" % len(tst_unique_tgts))
    logging.info("Tst unique lemmas: %d" % len(tst_unique_lemmas))
    logging.info("Tst unique taglists: %d" % len(tst_tagsets_dict))
    
    logging.info("seen_dev_tgt: %d" % seen_dev_tgt)
    logging.info("seen_dev_lemma: %d" % seen_dev_lemma)
    logging.info("seen_dev_taglist: %d" % seen_dev_taglist)
    logging.info("seen_tst_tgt: %d" % seen_tst_tgt)
    logging.info("seen_tst_lemma: %d" % seen_tst_lemma)
    logging.info("seen_tst_taglist: %d" % seen_tst_taglist)



    trnbatches = prepare_batches_with_no_pad(trainset,batchsize=BATCHSIZE)
    devbatches = prepare_batches_with_no_pad(devset, batchsize=BATCHSIZE)
    tstbatches = prepare_batches_with_no_pad_wlemmas(tstset, batchsize=1)
    _trnbatches = None
    dictmeta = []
    for tagkey,tagvalues in trainset.tagsvocab.vocabs.items():
        dictmeta.append(len(tagvalues.id2tag))

    ## load weights from ae
    #ae = AE()
    #ae.load_state_dict(torch.load("results/ae/1/ae.pt"))
    #zdict = torch.load('results/ae/1/2000trnwords_z.pt')
    #model.encoder.layers = ae.encoder.layers
    #model.down_to_z_layer = ae.down_to_z_layer
    #for idx, tensor in zdict.items():
    #    model.quantizer.embeddings.weight.data[idx] =  tensor
    #model.decoder.layers = ae.decoder.layers

    vocabsize = len(trainset.charvocab.char2id)
    # model
    args.mname = 'msvae' 
    model_init = uniform_initializer(0.01)
    emb_init = uniform_initializer(0.1)
    args.ni = 300; args.nz = 150; 
    args.enc_nh = 256; args.dec_nh = 256
    args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
    args.dec_dropout_in = 0.5; 
    args.device = 'cuda'
    args.update_temp = 2000

    model = MSVAE(args, trainset.charvocab, dictmeta, model_init, emb_init)
    model.to(device)
    
    opt = optim.Adam(model.parameters(), lr=LR,  betas=(0.5, 0.999))
    logging.info("trnID: %s" % str(TRN_ID))
    logging.info("batchsize: %d" % BATCHSIZE)
    logging.info("epochs: %d" % EPOCHS)
    logging.info("lr: %.5f" % LR)
    logging.info("KL_WEIGHT: %.5f" % KL_WEIGHT)
    logging.info("opt: %s", opt)
    logging.info(model)
    with open(path+"/charvocab.json", "w") as outfile:
        json.dump(trainset.charvocab.id2char, outfile, ensure_ascii=False, indent=4)

    for tag,vocab in trainset.tagsvocab.vocabs.items():
        with open(path+"/"+"vocab_"+tag+".json", "w") as outfile:
            json.dump(vocab.id2tag, outfile, ensure_ascii=False, indent=4)
    return model, opt, trnbatches, devbatches, tstbatches, trainset.charvocab, _trnbatches

def get_kth_element(dict,K):
    if len(dict) < K:
        K = len(dict)
    return [key for key in {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}.keys()][K-1]


def train(model, opt, trnbatches, devbatches, tstbatches, charvocab, _trnbatches):
    trnsize = sum([len(i[0]) for i in trnbatches.values()])
    devsize = sum([len(i[0]) for i in devbatches.values()])
    tstsize = sum([len(i[0]) for i in tstbatches.values()])

    logging.info("trnsize: %d" % trnsize)
    logging.info("devsize: %d" % devsize)
    logging.info("tstsize: %d" % tstsize)

    ## _trn
    kl_passed_epc = 0
    tmp = 1.0
    update_ind = 0
    for epc in range(EPOCHS):
        logging.info("")
        logging.info("#epc: %d" % epc)
        ## trn
        model.train()
        keys = list(trnbatches.keys())
        random.shuffle(keys)

        if epc >= KL_START_EPC:
            kl_passed_epc += 1
            epc_KL = min((kl_passed_epc/KL_ANNEAL_EPC)*KL_WEIGHT,KL_WEIGHT)
        else: 
            epc_KL = 0
        logging.info("epc:%d, epc_KL:  %.6f"%(epc,epc_KL))

        epoch_loss = 0; epoch_num_tokens = 0; 
        epoch_tag_total_tokens = 0; epoch_tag_correct= 0; 
        epoch_labeled_pred_loss = 0; epoch_labeled_recon_loss = 0; 
        epoch_labeled_kl_loss = 0; 
        epoch_labeled_num_tokens = 0
        epoch_labeled_reinflect_recon_acc = 0

        for bid in keys:
            batch = trnbatches[bid]
            tgt,tags = batch
            opt.zero_grad()
            if update_ind % args.update_temp == 0:
                tmp = get_temp(update_ind)
            kl_weight = get_kl_weight(update_ind, 0.2, 150000.0)
            update_ind +=1
            lx_src = tgt
            lx_tgt = tgt
            # (batchsize)
            loss_l, labeled_pred_loss, tag_correct, tag_total, labeled_recon_loss,  labeled_kl_loss, labeled_reinflect_recon_acc = model.loss_l(lx_src, tags, lx_tgt, epc_KL, tmp)
            epoch_labeled_num_tokens +=  torch.sum(lx_tgt[:,1:] !=0).item()
            epoch_tag_correct += tag_correct
            epoch_tag_total_tokens += tag_total
            epoch_labeled_pred_loss += labeled_pred_loss.item()
            epoch_labeled_recon_loss += labeled_recon_loss.sum().item()
            epoch_labeled_kl_loss += labeled_kl_loss.sum().item()
            epoch_labeled_reinflect_recon_acc  += labeled_reinflect_recon_acc
            batch_loss = loss_l.mean()
            batch_loss.backward()
            opt.step()
            epoch_loss += loss_l.sum().item()
        numwords = trnsize
        loss = epoch_loss / numwords  
        labeled_pred_loss = epoch_labeled_pred_loss/ epoch_tag_total_tokens
        labeled_pred_acc  = epoch_tag_correct/ epoch_tag_total_tokens
        labeled_recon_loss = epoch_labeled_recon_loss / epoch_labeled_num_tokens
        labeled_kl_loss = epoch_labeled_kl_loss / numwords
        labeled_reinflect_recon_acc = epoch_labeled_reinflect_recon_acc / epoch_labeled_num_tokens

        ## log            
        ##loss per token
        logging.info('epoch: %.1d, kl_weight: %.2f, tmp: %.2f' % (epc, kl_weight, tmp))
        logging.info('Trn--- loss: %.4f, labeled_pred_loss: %.4f, labeled_pred_acc: %.4f, labeled_recon_loss: %.4f, labeled_kl_loss: %.4f, labeled_reinflect_recon_acc: %.4f' % (loss, labeled_pred_loss, labeled_pred_acc, labeled_recon_loss,  labeled_kl_loss,  labeled_reinflect_recon_acc))
        #-end of trn

        ## dev (teacher forcing)
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_tag_total_tokens = 0; epoch_tag_correct= 0; 
        epoch_labeled_pred_loss = 0; epoch_labeled_recon_loss = 0
        epoch_labeled_num_tokens = 0
        epoch_labeled_kl_loss = 0
        epoch_labeled_reinflect_recon_acc = 0
        model.eval()
        for bid, batch in devbatches.items():
            tgt,tags = batch
            lx_src = tgt
            lx_tgt = tgt
            loss, labeled_pred_loss, tag_correct, tag_total, labeled_recon_loss, labeled_kl_loss, labeled_reinflect_recon_acc  = model.loss_l(lx_src, tags, lx_tgt, epc_KL, tmp)
            epoch_tag_correct += tag_correct
            epoch_tag_total_tokens += tag_total
            epoch_labeled_num_tokens +=  torch.sum(lx_tgt[:,1:] !=0).item()
            epoch_loss       += loss.sum().item()
            epoch_labeled_pred_loss += labeled_pred_loss.item()
            epoch_labeled_recon_loss += labeled_recon_loss.sum().item()
            epoch_labeled_kl_loss += labeled_kl_loss.sum().item()
            epoch_labeled_reinflect_recon_acc  += labeled_reinflect_recon_acc
        loss = epoch_loss / numwords 
        recon = epoch_recon_loss / numwords 
        labeled_pred_acc = epoch_tag_correct/ epoch_tag_total_tokens
        labeled_pred_loss = epoch_labeled_pred_loss / epoch_tag_total_tokens
        labeled_recon_loss = epoch_labeled_recon_loss / epoch_labeled_num_tokens
        labeled_kl_loss = epoch_labeled_kl_loss / numwords
        labeled_reinflect_recon_acc = epoch_labeled_reinflect_recon_acc / epoch_labeled_num_tokens
        logging.info('Tst--- loss: %.4f, labeled_pred_loss: %.4f, labeled_pred_acc: %.4f, labeled_recon_loss: %.4f, labeled_kl_loss: %.4f,  labeled_reinflect_recon_acc: %.4f' % ( loss, labeled_pred_loss, labeled_pred_acc, labeled_recon_loss,  labeled_kl_loss, labeled_reinflect_recon_acc))


        ## Copy & Reinflect tst with one-step at a time
        out_df = pd.DataFrame({})
        i=0
        exact_match_tst_acc = 0
        for bid, batch in tstbatches.items():
            tgt,tags,lemma = batch
            lemmas       = decode_batch(lemma[:,1:], charvocab)[0]
            gold_decoded = decode_batch(tgt[:,1:], charvocab)[0]
            if gold_decoded == "Hakk'ın rahmetine kavuşuyor olacak mıymışsınız</s>":
                _tags= tags
            out_df.at[i, "lemma"] = lemmas
            out_df.at[i, "gold"] = gold_decoded
            ## Copy the tgt
            reinflected_form = model.decode(tgt,tags,tmp)
            out_df.at[i, "COPY_pred"] = reinflected_form
            i+=1
            if gold_decoded == reinflected_form:
                out_df.at[i, "exact_match"] = 1
                exact_match_tst_acc+=1
        logging.info("exact_match_tst_acc: %.3f" % (exact_match_tst_acc/tstsize))
        ## Sample words with different lemmas
        out_df = pd.DataFrame({})
        for i in range(50):
            out_df.at[i, "sampled"] = model.sample(_tags)
        out_df.to_csv(path+'/samples_epc'+str(epc)+'.csv')

model, opt, trn, dev, tst, charvocab, _trnbatches = config()
train(model, opt, trn, dev, tst, charvocab, _trnbatches)