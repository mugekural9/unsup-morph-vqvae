import torch, os, logging, shutil, random, json
import pandas as pd
from torch import optim
from data.datareader import *
from data.dataset import MorphDataset
from ae.model import AE
from ae.util import accuracy_on_batch, decode_batch

from torch.utils.tensorboard import SummaryWriter
from ae.test import test

TRN_ID = 2
path = 'results/ae/'+str(TRN_ID)
try:
    shutil.rmtree(path)
except OSError as error:
    print(error)  

try:
    os.mkdir(path)
except OSError as error:
    print(error)  


#random.seed(0)
BATCHSIZE = 64
EPOCHS = 200
LR = 0.0005
writer = SummaryWriter(comment="_AE_TRNID_"+str(TRN_ID))
device = "cuda" if torch.cuda.is_available() else "cpu"


def config():
    logging.basicConfig(handlers=[
            logging.FileHandler("results/ae/"+str(TRN_ID)+"/training_ae.log"),
            logging.StreamHandler()],
            format='%(asctime)s - %(message)s', level=logging.INFO)
    trainset = MorphDataset("data/tur_large.train")
    devset = MorphDataset("data/tur.dev", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
    trnbatches = prepare_batches_with_no_pad(trainset,batchsize=BATCHSIZE)
    devbatches = prepare_batches_with_no_pad(devset, batchsize=BATCHSIZE)
    tstbatches = prepare_batches_with_no_pad(devset, batchsize=1)

    model = AE()
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=LR,  betas=(0.5, 0.999))
    logging.info("trnID: %d" % TRN_ID)
    logging.info("batchsize: %d" % BATCHSIZE)
    logging.info("epochs: %d" % EPOCHS)
    logging.info("lr: %.5f" % LR)
    logging.info("opt: %s", opt)

    logging.info(model)
    with open("results/ae/"+str(TRN_ID)+"/charvocab.json", "w") as outfile:
        json.dump(trainset.charvocab.id2char, outfile, ensure_ascii=False, indent=4)

    for tag,vocab in trainset.tagsvocab.vocabs.items():
        with open("results/ae/"+str(TRN_ID)+"/"+tag+"_vocab.json", "w") as outfile:
            json.dump(vocab.id2tag, outfile, ensure_ascii=False, indent=4)
    return model, opt, trnbatches, devbatches, tstbatches, trainset.charvocab


def train(model, opt, trnbatches, devbatches, tstbatches, charvocab):
    trnsize = sum([len(i) for i in trnbatches.values()])
    devsize = sum([len(i) for i in devbatches.values()])
    tstsize = sum([len(i) for i in tstbatches.values()])

    logging.info("trnsize: %d" % trnsize)
    logging.info("devsize: %d" % devsize)
    
    best_exact_match = 0
    for epc in range(EPOCHS):
        logging.info("")
        logging.info("#epc: %d" % epc)

        epc_loss = dict({'trn':0, 'dev':0})
        epc_nonpaddedtokens = dict({'trn':0, 'dev':0})
        epc_exactmatch = dict({'trn':0, 'dev':0})
        epc_tokensmatch = dict({'trn':0, 'dev':0})

        ## trn
        model.train()
        keys = list(trnbatches.keys())
        random.shuffle(keys)
        for bid in keys:
            batch = trnbatches[bid]
            opt.zero_grad()
            batchloss, nonpadded_tokens, predtokens = model(batch)
            optloss = (batchloss/nonpadded_tokens)
            optloss.backward()
            opt.step()
            epc_loss['trn']+= batchloss.item()
            epc_nonpaddedtokens['trn'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, batch[:,1:])
            epc_exactmatch['trn'] += exact_match
            epc_tokensmatch['trn'] += tokens_match

        ## dev
        model.eval()
        for bid, batch in devbatches.items():
            batchloss, nonpadded_tokens, predtokens = model(batch)
            epc_loss['dev']+= batchloss.item()
            epc_nonpaddedtokens['dev'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, batch[:,1:])
            epc_exactmatch['dev'] += exact_match
            epc_tokensmatch['dev'] += tokens_match


        ## tst
        out_df = pd.DataFrame({})
        i=0
        for bid, batch in tstbatches.items():
            batchloss, nonpadded_tokens, predtokens  = model(batch)
            predtokens, _ = model.decode(batch)
            gold_decoded_batches = decode_batch(batch[:,1:], charvocab)
            pred_decoded_batches = decode_batch(predtokens, charvocab)
            for g,p in zip(gold_decoded_batches, pred_decoded_batches):
                out_df.at[i, "gold"] = g
                out_df.at[i, "pred"] = p
                if g == p:
                    out_df.at[i, "exact_match"] = 1
                else:
                    out_df.at[i, "exact_match"] = 0
                i+=1
        out_df.to_csv(path+'/preds_'+str(epc)+'_.csv')


        ## log            
        logging.info("TRN || epcloss: %.4f" % (epc_loss['trn']/ epc_nonpaddedtokens['trn']))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['trn']/epc_nonpaddedtokens['trn']), (epc_exactmatch['trn']/trnsize)))
        logging.info("DEV || epcloss: %.4f" % (epc_loss['dev']/epc_nonpaddedtokens['dev']))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['dev']/epc_nonpaddedtokens['dev']), (epc_exactmatch['dev']/devsize)))

        writer.add_scalar('loss/train', epc_loss['trn']/ epc_nonpaddedtokens['trn'], epc)
        writer.add_scalar('loss/dev',   epc_loss['dev']/ epc_nonpaddedtokens['dev'], epc)
        writer.add_scalar('token_match/train',epc_tokensmatch['trn']/ epc_nonpaddedtokens['trn'], epc)
        writer.add_scalar('token_match/dev',  epc_tokensmatch['dev']/ epc_nonpaddedtokens['dev'], epc)
       
        writer.add_scalar('exact_match/train',epc_exactmatch['trn']/trnsize, epc)
        writer.add_scalar('exact_match/dev',  epc_exactmatch['dev']/devsize, epc)
        
        ## save
        if  epc_exactmatch['dev'] > best_exact_match:
            logging.info("    || BEST exact match: %.3f" % (epc_exactmatch['dev']/devsize))
            torch.save(model.state_dict(), "results/ae/"+str(TRN_ID)+"/ae.pt")
            logging.info("    || saved model")
            best_exact_match = epc_exactmatch['dev']
            best_epc = epc
    logging.info("Training is over, saved model with best dev exact match: %.3f (epc %d)" % (best_exact_match/devsize, best_epc))


model, opt, trn, dev, tst, charvocab = config()
train(model, opt, trn, dev, tst, charvocab)