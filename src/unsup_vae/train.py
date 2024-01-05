import torch, os, logging, shutil, random, json
import pandas as pd
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from dataset.morph_dataset import MorphDataset
from dataset.datareader import *
from unsup_vae.model import VAE
from util import accuracy_on_batch, decode_batch

BATCHSIZE = 64
EPOCHS = 200
LR = 0.0005
KL_WEIGHT = 1.0
KL_ANNEAL_EPC = 1
KL_START_EPC = 2
ENC_NH = 256
DEC_NH = 512
Z_LEMMA_NH = 32

NUM_CODEBOOK = 2
NUM_CODEBOOK_ENTRIES = 10
BIDIRECTIONAL = True

DEC_DROPOUT = 0.2
INPUT_EMB_DIM = 256
TRN_ID = "BI"+str(BIDIRECTIONAL)+"_UtrNOUN_zLEM"+str(Z_LEMMA_NH)+ "_decnh"+str(DEC_NH)+"_kl"+str(KL_WEIGHT)+"_epc"+str(KL_ANNEAL_EPC)+"_strt"+str(KL_START_EPC)+"_decdo"+str(DEC_DROPOUT)+"_inpemb"+str(INPUT_EMB_DIM)+"_bsize"+str(BATCHSIZE)
path = 'results/vae/unsup/'+str(TRN_ID)
try:
    shutil.rmtree(path)
except OSError as error:
    print(error)  

try:
    os.mkdir(path)
except OSError as error:
    print(error)  

#random.seed(0)
writer = SummaryWriter(comment="_VAE_TRNID_"+str(TRN_ID))
device = "cuda" if torch.cuda.is_available() else "cpu"


import json


def config():
    logging.basicConfig(handlers=[
            logging.FileHandler(path+"/training_vqvae.log"),
            logging.StreamHandler()],
            format='%(asctime)s - %(message)s', level=logging.INFO)
    trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur/tur_filtered_duplicates_merged_stdata_NOUNS_shuffled")
    devset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_NOUNS", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
    tstset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_NOUNS", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)

    maxtrnsize = 512000
    trainset.lemmas = trainset.lemmas[:maxtrnsize]
    trainset.tgts = trainset.tgts[:maxtrnsize]
    trainset.tagslist = trainset.tagslist[:maxtrnsize]
  
    maxdevsize = maxtrnsize
    devset.lemmas = devset.lemmas[:maxdevsize]
    devset.tgts = devset.tgts[:maxdevsize]
    devset.tagslist = devset.tagslist[:maxdevsize]
      
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
    devbatches = prepare_batches_with_no_pad(tstset, batchsize=BATCHSIZE)
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
    model = VAE(vocabsize,  ENC_NH, DEC_NH, Z_LEMMA_NH, DEC_DROPOUT, INPUT_EMB_DIM, BIDIRECTIONAL)

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
        with open(path+"/"+tag+"_vocab.json", "w") as outfile:
            json.dump(vocab.id2tag, outfile, ensure_ascii=False, indent=4)
    return model, opt, trnbatches, devbatches, tstbatches, trainset.charvocab


def train(model, opt, trnbatches, devbatches, tstbatches, charvocab):
    trnsize = sum([len(i[0]) for i in trnbatches.values()])
    devsize = sum([len(i[0]) for i in devbatches.values()])
    tstsize = sum([len(i[0]) for i in tstbatches.values()])

    logging.info("trnsize: %d" % trnsize)
    logging.info("devsize: %d" % devsize)
    logging.info("tstsize: %d" % tstsize)

    best_exact_match = 0
    kl_passed_epc = 0
    for epc in range(EPOCHS):
        logging.info("")
        logging.info("#epc: %d" % epc)

        epc_loss = dict({'trn':0, 'dev':0})
        epc_KL_loss = dict({'trn':0, 'dev':0})
        epc_nonpaddedtokens = dict({'trn':0, 'dev':0})
        epc_exactmatch = dict({'trn':0, 'dev':0})
        epc_tokensmatch = dict({'trn':0, 'dev':0})

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
        for bid in keys:
            batch = trnbatches[bid]
            tgt,tags = batch
            opt.zero_grad()
            batchloss, nonpadded_tokens, predtokens, KL = model(tgt)
            #optimize for per token nll + KL per word
            optloss = (batchloss/nonpadded_tokens) +  epc_KL * (KL/tgt.size(0))
            optloss.backward()
            opt.step()
            epc_loss['trn']+= batchloss.item()
            epc_nonpaddedtokens['trn'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, tgt[:,1:])
            epc_exactmatch['trn'] += exact_match
            epc_tokensmatch['trn'] += tokens_match
            epc_KL_loss['trn'] += KL
        #-end of trn
 
        ## dev (teacher forcing)
        model.eval()
        for bid, batch in devbatches.items():
            tgt,tags = batch
            batchloss, nonpadded_tokens, predtokens, KL= model(tgt)
            epc_loss['dev']+= batchloss.item()
            epc_nonpaddedtokens['dev'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, tgt[:,1:])
            epc_exactmatch['dev'] += exact_match
            epc_tokensmatch['dev'] += tokens_match
            epc_KL_loss['dev'] += KL
         


        ## Copy dev with one-step at a time
        out_df = pd.DataFrame({})
        exact_match_tst_acc=0
        i=0
        for bid, batch in tstbatches.items():
            tgt,tags,lemma = batch
            predtokens = model.decode(tgt)
            lemmas = decode_batch(lemma[:,1:], charvocab)
            gold_decoded = decode_batch(tgt[:,1:], charvocab)
            pred_decoded = decode_batch(predtokens, charvocab)
            for g,p,l in zip(gold_decoded, pred_decoded, lemmas):
                if g == p:
                    out_df.at[i, "exact_match"] = 1
                    exact_match_tst_acc+=1
                else:
                    out_df.at[i, "exact_match"] = 0
                out_df.at[i, "lemma"] = l
                out_df.at[i, "gold"] = g
                out_df.at[i, "pred"] = p
                i+=1
            if i>500:
                break
            out_df.to_csv(path+'/tst_copies_'+str(epc)+'.csv')
        tst_acc = exact_match_tst_acc / i
        logging.info("exact_match_tst_copy_acc: %.3f" % tst_acc)
        writer.add_scalar('exact_match_tst_copy_acc', tst_acc, epc)


        ## Sample words with different lemmas
        out_df = pd.DataFrame({})
        for i in range(50):
            predtokens = model.sample()
            decoded_batches = decode_batch(predtokens, charvocab)
            out_df.at[i, "sampled"] = decoded_batches[0]
        out_df.to_csv(path+'/samples_epc'+str(epc)+'.csv')



        ## log            
        ##loss per token
        logging.info("TRN || epcloss: %.4f" % (epc_loss['trn']/ epc_nonpaddedtokens['trn']))
        logging.info("    || KL: %.4f" % (epc_KL_loss['trn']/trnsize))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['trn']/epc_nonpaddedtokens['trn']), (epc_exactmatch['trn']/trnsize)))
        logging.info("DEV || epcloss: %.4f" % (epc_loss['dev']/epc_nonpaddedtokens['dev']))
        logging.info("    || KL: %.4f" % (epc_KL_loss['dev']/devsize))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['dev']/epc_nonpaddedtokens['dev']), (epc_exactmatch['dev']/devsize)))
        if  epc_exactmatch['dev'] > best_exact_match:
            logging.info("    || BEST exact match (teacher forcing): %.3f" % (epc_exactmatch['dev']/devsize))
            best_exact_match = epc_exactmatch['dev']
            best_epc = epc
        
        ## save
        #if epc>30 and epc%3==0:
            #torch.save(model.state_dict(), "results/vqvae/unsup/"+str(TRN_ID)+"/vqvae_"+str(epc)+".pt")
            #logging.info("    || saved model")

        writer.add_scalar('loss/train', epc_loss['trn']/ epc_nonpaddedtokens['trn'], epc)
        writer.add_scalar('loss/dev',   epc_loss['dev']/ epc_nonpaddedtokens['dev'], epc)
        writer.add_scalar('KL_loss/trn',   epc_KL_loss['trn']/trnsize, epc)
        writer.add_scalar('KL_loss/dev',   epc_KL_loss['dev']/devsize, epc)
        writer.add_scalar('token_match/trn',epc_tokensmatch['trn']/ epc_nonpaddedtokens['trn'], epc)
        writer.add_scalar('token_match/dev',  epc_tokensmatch['dev']/ epc_nonpaddedtokens['dev'], epc)
        writer.add_scalar('exact_match/trn',epc_exactmatch['trn']/trnsize, epc)
        writer.add_scalar('exact_match/dev',  epc_exactmatch['dev']/devsize, epc)
    logging.info("Training is over, saved model with best dev exact match: %.3f (epc %d)" % (best_exact_match/devsize, best_epc))


model, opt, trn, dev, tst, charvocab = config()
train(model, opt, trn, dev, tst, charvocab)