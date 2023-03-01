import torch, os, logging, shutil, random, json
import pandas as pd
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from data.dataset import MorphDataset
from data.datareader import *
from vae.model import VAE
from vae.test import test
from vae.util import accuracy_on_batch, decode_batch

TRN_ID = 1
path = 'results/vae/'+str(TRN_ID)
try:
    shutil.rmtree(path)
except OSError as error:
    print(error)  

try:
    os.mkdir(path)
except OSError as error:
    print(error)  


#random.seed(0)
BATCHSIZE = 32
EPOCHS = 60
LR = 0.0005
KL_START = 0
KL_START_EPC = 20
KL_MAX = 0.1
WARM_UP = 500
KL_ANNEAL = True
writer = SummaryWriter(comment="_VAE_TRNID_"+str(TRN_ID))
device = "cuda" if torch.cuda.is_available() else "cpu"


def config():
    logging.basicConfig(handlers=[
            logging.FileHandler("results/vae/"+str(TRN_ID)+"/training_vae.log"),
            logging.StreamHandler()],
            format='%(asctime)s - %(message)s', level=logging.INFO)
    trainset = MorphDataset("data/tur_large.train")
    devset = MorphDataset("data/tur.dev", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
    trnbatches = prepare_batches_with_no_pad(trainset,batchsize=BATCHSIZE)
    devbatches = prepare_batches_with_no_pad(devset, batchsize=BATCHSIZE)
    tstbatches = prepare_batches_with_no_pad(devset, batchsize=1)
    model = VAE()
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    logging.info("trnID: %d" % TRN_ID)
    logging.info("batchsize: %d" % BATCHSIZE)
    logging.info("epochs: %d" % EPOCHS)
    logging.info("lr: %.5f" % LR)
    logging.info(model)
    with open("results/vae/"+str(TRN_ID)+"/charvocab.json", "w") as outfile:
        json.dump(trainset.charvocab.id2char, outfile, ensure_ascii=False, indent=4)

    for tag,vocab in trainset.tagsvocab.vocabs.items():
        with open("results/vae/"+str(TRN_ID)+"/"+tag+"_vocab.json", "w") as outfile:
            json.dump(vocab.id2tag, outfile, ensure_ascii=False, indent=4)
    return model, opt, trnbatches, devbatches, tstbatches, trainset.charvocab


def train(model, opt, trnbatches, devbatches, tstbatches, charvocab):
    trnsize = sum([len(i) for i in trnbatches.values()])
    devsize = sum([len(i) for i in devbatches.values()])
    tstsize = sum([len(i) for i in tstbatches.values()])

    logging.info("trnsize: %d" % trnsize)
    logging.info("devsize: %d" % devsize)
    
    best_exact_match = 0
    kl_weight = KL_START
    anneal_rate = (1.0 - KL_START) / (WARM_UP * len(trnbatches))
    for epc in range(EPOCHS):
        logging.info("")
        logging.info("#epc: %d" % epc)

        epc_loss = dict({'trn':0, 'dev':0})
        epc_kl_loss = dict({'trn':0, 'dev':0})
        epc_nonpaddedtokens = dict({'trn':0, 'dev':0})
        epc_exactmatch = dict({'trn':0, 'dev':0})
        epc_tokensmatch = dict({'trn':0, 'dev':0})

        ## trn
        model.train()
        keys = list(trnbatches.keys())
        random.shuffle(keys)
        for bid in keys:
            if epc>KL_START_EPC and KL_ANNEAL:
                kl_weight = min(KL_MAX, kl_weight + anneal_rate)
            batch = trnbatches[bid]
            opt.zero_grad()
            batchloss, nonpadded_tokens, predtokens, KL = model(batch)
            #optimize for per token nll + KL per word
            optloss = (batchloss/nonpadded_tokens) + (kl_weight * (KL/batch.size(0)))
            optloss.backward()
            opt.step()
            epc_loss['trn']+= batchloss.item()
            epc_nonpaddedtokens['trn'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, batch[:,1:])
            epc_exactmatch['trn'] += exact_match
            epc_tokensmatch['trn'] += tokens_match
            epc_kl_loss['trn'] += KL

        ## dev
        model.eval()
        for bid, batch in devbatches.items():
            batchloss, nonpadded_tokens, predtokens, KL = model(batch)
            epc_loss['dev']+= batchloss.item()
            epc_nonpaddedtokens['dev'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, batch[:,1:])
            epc_exactmatch['dev'] += exact_match
            epc_tokensmatch['dev'] += tokens_match
            epc_kl_loss['dev'] += KL

        ## tst
        out_df = pd.DataFrame({})
        i=0
        for bid, batch in tstbatches.items():
            predtokens = model.decode(batch)
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
            total_exact_match =  sum(out_df.exact_match==1)
        out_df.to_csv(path+'/preds_'+str(epc)+'_.csv')


        ## log            
        ##loss per token
        logging.info("TRN || epcloss: %.4f" % (epc_loss['trn']/ epc_nonpaddedtokens['trn']))
        logging.info("    || KL: %.4f" % (epc_kl_loss['trn']/trnsize))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['trn']/epc_nonpaddedtokens['trn']), (epc_exactmatch['trn']/trnsize)))
        logging.info("    || kl_weight: %.4f" % kl_weight)
        logging.info("DEV || epcloss: %.4f" % (epc_loss['dev']/epc_nonpaddedtokens['dev']))
        logging.info("    || KL: %.4f" % (epc_kl_loss['dev']/devsize))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['dev']/epc_nonpaddedtokens['dev']), (epc_exactmatch['dev']/devsize)))

        writer.add_scalar('loss/train', epc_loss['trn']/ epc_nonpaddedtokens['trn'], epc)
        writer.add_scalar('loss/dev',   epc_loss['dev']/ epc_nonpaddedtokens['dev'], epc)
        writer.add_scalar('KL_loss/trn',   epc_kl_loss['trn']/trnsize, epc)
        writer.add_scalar('KL_loss/dev',   epc_kl_loss['dev']/devsize, epc)
        writer.add_scalar('KL_weight', kl_weight, epc)
        writer.add_scalar('token_match/train',epc_tokensmatch['trn']/ epc_nonpaddedtokens['trn'], epc)
        writer.add_scalar('token_match/dev',  epc_tokensmatch['dev']/ epc_nonpaddedtokens['dev'], epc)
       
        writer.add_scalar('exact_match/train',epc_exactmatch['trn']/trnsize, epc)
        writer.add_scalar('exact_match/dev',  epc_exactmatch['dev']/devsize, epc)

        ## save
        if epc>20 and epc % 3 ==0:
            torch.save(model.state_dict(), "results/vae/"+str(TRN_ID)+"/vae_epc"+str(epc)+".pt")
        if  epc_exactmatch['dev'] > best_exact_match:
            logging.info("    || BEST exact match: %.3f" % (epc_exactmatch['dev']/devsize))
            torch.save(model.state_dict(), "results/vae/"+str(TRN_ID)+"/vae.pt")
            logging.info("    || saved model")
            best_exact_match = epc_exactmatch['dev']
            best_epc = epc
    logging.info("Training is over, saved model with best dev exact match: %.3f (epc %d)" % (best_exact_match/devsize, best_epc))


model, opt, trn, dev, tst, charvocab = config()
train(model, opt, trn, dev, tst, charvocab)