import torch, os, logging, shutil, random, json
import pandas as pd
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from data.dataset import MorphDataset
from data.datareader import *
from ae.model import AE
from vqvae.model_2 import VQVAE
from vqvae.util import accuracy_on_batch, decode_batch

TRN_ID = 7
path = 'results/vqvae/'+str(TRN_ID)
try:
    shutil.rmtree(path)
except OSError as error:
    print(error)  

try:
    os.mkdir(path)
except OSError as error:
    print(error)  


#random.seed(0)
BATCHSIZE = 256
EPOCHS = 1000
LR = 0.0005
writer = SummaryWriter(comment="_VQVAE_TRNID_"+str(TRN_ID))
device = "cuda" if torch.cuda.is_available() else "cpu"




def config():
    logging.basicConfig(handlers=[
            logging.FileHandler("results/vqvae/"+str(TRN_ID)+"/training_vqvae.log"),
            logging.StreamHandler()],
            format='%(asctime)s - %(message)s', level=logging.INFO)
    trainset = MorphDataset("data/tur_large.train")
    devset = MorphDataset("data/tur.dev", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)

    maxtrnsize = 10000
    trainset.lemmas = trainset.lemmas[:maxtrnsize]
    trainset.tgts = trainset.tgts[:maxtrnsize]
    trainset.tagslist = trainset.tagslist[:maxtrnsize]
    
    maxdevsize = maxtrnsize
    devset.lemmas = trainset.lemmas[:maxdevsize]
    devset.tgts = trainset.tgts[:maxdevsize]
    devset.tagslist = trainset.tagslist[:maxdevsize]

    trnbatches = prepare_batches_with_no_pad(trainset,batchsize=BATCHSIZE)
    devbatches = prepare_batches_with_no_pad(devset, batchsize=BATCHSIZE)
    tstbatches = prepare_batches_with_no_pad(devset, batchsize=1)
    
    model = VQVAE()

    ## load weights from ae
    ae = AE()
    ae.load_state_dict(torch.load("results/ae/1/ae.pt"))
    zdict_lemma = torch.load('results/ae/1/2000trnwords_z.pt')
    zdict_affix = torch.load('results/ae/1/100trnwords_z.pt')

    model.encoder.layers = ae.encoder.layers
    model.down_to_z_layer = ae.down_to_z_layer
    for idx, tensor in zdict_lemma.items():
        model.quantizer_lemma.embeddings.weight.data[idx] =  tensor[:,:,:16]
    
    for idx, tensor in zdict_affix.items():
        model.quantizer_affix.embeddings.weight.data[idx] =  tensor[:,:,16:]
    
    #model.decoder.layers = ae.decoder.layers
    model.to(device)
    
    opt = optim.Adam(model.parameters(), lr=LR,  betas=(0.5, 0.999))
    logging.info("trnID: %d" % TRN_ID)
    logging.info("batchsize: %d" % BATCHSIZE)
    logging.info("epochs: %d" % EPOCHS)
    logging.info("lr: %.5f" % LR)
    logging.info("opt: %s", opt)

    logging.info(model)
    with open("results/vqvae/"+str(TRN_ID)+"/charvocab.json", "w") as outfile:
        json.dump(trainset.charvocab.id2char, outfile, ensure_ascii=False, indent=4)

    for tag,vocab in trainset.tagsvocab.vocabs.items():
        with open("results/vqvae/"+str(TRN_ID)+"/"+tag+"_vocab.json", "w") as outfile:
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
        epc_q_loss_lemma = dict({'trn':0, 'dev':0})
        epc_q_loss_affix = dict({'trn':0, 'dev':0})

        epc_q_indices_lemma = dict({'trn': defaultdict(lambda:0), 'dev':defaultdict(lambda:0)})
        epc_q_indices_affix = dict({'trn': defaultdict(lambda:0), 'dev':defaultdict(lambda:0)})

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
            batchloss, nonpadded_tokens, predtokens, Q, quantized_indices = model(batch)
            Q_lemma, Q_affix = Q
            quantized_indices_lemma, quantized_indices_affix = quantized_indices

            #optimize for per token nll + Q per word
            optloss = (batchloss/nonpadded_tokens) +  (Q_lemma/batch.size(0)) + (Q_affix/batch.size(0))
            optloss.backward()
            opt.step()
            epc_loss['trn']+= batchloss.item()
            epc_nonpaddedtokens['trn'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, batch[:,1:])
            epc_exactmatch['trn'] += exact_match
            epc_tokensmatch['trn'] += tokens_match
          
            epc_q_loss_lemma['trn'] += Q_lemma
            epc_q_loss_affix['trn'] += Q_affix


            qlist_lemma = quantized_indices_lemma.squeeze(1).tolist()
            qlist_affix = quantized_indices_affix.squeeze(1).tolist()
            
            for q in set(qlist_lemma):
                epc_q_indices_lemma['trn'][q] +=qlist_lemma.count(q)
            for q in set(qlist_affix):
                epc_q_indices_affix['trn'][q] +=qlist_affix.count(q)

        ## dev
        model.eval()
        for bid, batch in devbatches.items():
            batchloss, nonpadded_tokens, predtokens, Q, quantized_indices = model(batch)
            Q_lemma, Q_affix = Q
            quantized_indices_lemma, quantized_indices_affix = quantized_indices

            
            epc_loss['dev']+= batchloss.item()
            epc_nonpaddedtokens['dev'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, batch[:,1:])
            epc_exactmatch['dev'] += exact_match
            epc_tokensmatch['dev'] += tokens_match

            epc_q_loss_lemma['dev'] += Q_lemma
            epc_q_loss_affix['dev'] += Q_affix

            qlist_lemma = quantized_indices_lemma.squeeze(1).tolist()
            qlist_affix = quantized_indices_affix.squeeze(1).tolist()
            
            for q in set(qlist_lemma):
                epc_q_indices_lemma['dev'][q] +=qlist_lemma.count(q)
            for q in set(qlist_affix):
                epc_q_indices_affix['dev'][q] +=qlist_affix.count(q)

        '''
        ## tst
        out_df = pd.DataFrame({})
        i=0
        for bid, batch in tstbatches.items():
            _, _, _, _,  quantized_indices = model(batch)
            quantized_indices_lemma, quantized_indices_affix = quantized_indices
            predtokens = model.decode(batch)
            gold_decoded_batches = decode_batch(batch[:,1:], charvocab)
            pred_decoded_batches = decode_batch(predtokens, charvocab)
            for g,p in zip(gold_decoded_batches, pred_decoded_batches):
                out_df.at[i, "gold"] = g
                out_df.at[i, "pred"] = p
                out_df.at[i, "codebook_entry_lemma"] = quantized_indices_lemma.item()
                out_df.at[i, "codebook_entry_affix"] = quantized_indices_affix.item()

                if g == p:
                    out_df.at[i, "exact_match"] = 1
                else:
                    out_df.at[i, "exact_match"] = 0
                i+=1
        out_df.to_csv(path+'/preds_'+str(epc)+'_.csv')
        '''

        ## log            
        ##loss per token
        logging.info("TRN || epcloss: %.4f" % (epc_loss['trn']/ epc_nonpaddedtokens['trn']))
        logging.info("    || Q_lemma: %.4f" % (epc_q_loss_lemma['trn']/trnsize))
        logging.info("    || Q_affix: %.4f" % (epc_q_loss_affix['trn']/trnsize))

      
        #logging.info("    || Q_indices: %s" % epc_q_indices['trn'])
        logging.info("    || Num_Q_indices_lemma: %s" % len(epc_q_indices_lemma['trn']))
        logging.info("    || Num_Q_indices_affix: %s" % len(epc_q_indices_affix['trn']))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['trn']/epc_nonpaddedtokens['trn']), (epc_exactmatch['trn']/trnsize)))


        logging.info("DEV || epcloss: %.4f" % (epc_loss['dev']/epc_nonpaddedtokens['dev']))
        logging.info("    || Q_lemma: %.4f" % (epc_q_loss_lemma['dev']/devsize))
        logging.info("    || Q_affix: %.4f" % (epc_q_loss_affix['dev']/devsize))
        
        logging.info("    || Num_Q_indices_lemma: %s" % len(epc_q_indices_lemma['dev']))
        logging.info("    || Num_Q_indices_affix: %s" % len(epc_q_indices_affix['dev']))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['dev']/epc_nonpaddedtokens['dev']), (epc_exactmatch['dev']/devsize)))


        writer.add_scalar('loss/train', epc_loss['trn']/ epc_nonpaddedtokens['trn'], epc)
        writer.add_scalar('loss/dev',   epc_loss['dev']/ epc_nonpaddedtokens['dev'], epc)
       
       
        writer.add_scalar('Q_loss_lemma/trn',   epc_q_loss_lemma['trn']/trnsize, epc)
        writer.add_scalar('Q_loss_lemma/dev',   epc_q_loss_lemma['dev']/devsize, epc)
        writer.add_scalar('Q_loss_affix/trn',   epc_q_loss_affix['trn']/trnsize, epc)
        writer.add_scalar('Q_loss_affix/dev',   epc_q_loss_affix['dev']/devsize, epc)

        writer.add_scalar('Q_NUM_lemma/trn',   len(epc_q_indices_lemma['trn']), epc)
        writer.add_scalar('Q_NUM_lemma/dev',   len(epc_q_indices_lemma['dev']), epc)
      
        writer.add_scalar('Q_NUM_affix/trn',   len(epc_q_indices_affix['trn']), epc)
        writer.add_scalar('Q_NUM_affix/dev',   len(epc_q_indices_affix['dev']), epc)

      
        writer.add_scalar('token_match/train',epc_tokensmatch['trn']/ epc_nonpaddedtokens['trn'], epc)
        writer.add_scalar('token_match/dev',  epc_tokensmatch['dev']/ epc_nonpaddedtokens['dev'], epc)
       
        writer.add_scalar('exact_match/train',epc_exactmatch['trn']/trnsize, epc)
        writer.add_scalar('exact_match/dev',  epc_exactmatch['dev']/devsize, epc)

        ## save
        #if epc>20 and epc % 3 ==0:
        #    torch.save(model.state_dict(), "results/vqvae/"+str(TRN_ID)+"/vqvae_epc"+str(epc)+".pt")
        if  epc_exactmatch['dev'] > best_exact_match:
            logging.info("    || BEST exact match: %.3f" % (epc_exactmatch['dev']/devsize))
            torch.save(model.state_dict(), "results/vqvae/"+str(TRN_ID)+"/vqvae.pt")
            logging.info("    || saved model")
            best_exact_match = epc_exactmatch['dev']
            best_epc = epc
    logging.info("Training is over, saved model with best dev exact match: %.3f (epc %d)" % (best_exact_match/devsize, best_epc))


model, opt, trn, dev, tst, charvocab = config()
train(model, opt, trn, dev, tst, charvocab)