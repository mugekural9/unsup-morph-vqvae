import torch, os, logging, shutil, random, json
import pandas as pd
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from dataset.morph_dataset import MorphDataset
from dataset.datareader import *
from unsup.model_v2 import VQVAE
from util import accuracy_on_batch, decode_batch

BATCHSIZE = 64
EPOCHS = 200
LR = 0.0005
KL_WEIGHT = 0.1
KL_ANNEAL_EPC = 10
KL_START_EPC = 5
ENC_NH = 256
DEC_NH = 350
Z_LEMMA_NH = 100
Z_TAG_NH = 128

NUM_CODEBOOK = 2
NUM_CODEBOOK_ENTRIES = 10
BIDIRECTIONAL = True

DEC_DROPOUT = 0.2
INPUT_EMB_DIM = 64
TRN_ID = "v2_BI"+str(BIDIRECTIONAL)+"_"+str(NUM_CODEBOOK)+"x"+str(NUM_CODEBOOK_ENTRIES)+"_UtrNOUN_zLEM"+str(Z_LEMMA_NH)+"_zTAG"+ str(Z_TAG_NH)+ "_decnh"+str(DEC_NH)+"_kl"+str(KL_WEIGHT)+"_epc"+str(KL_ANNEAL_EPC)+"_strt"+str(KL_START_EPC)+"_decdo"+str(DEC_DROPOUT)+"_inpemb"+str(INPUT_EMB_DIM)+"_bsize"+str(BATCHSIZE)
path = 'results/vqvae/unsup/'+str(TRN_ID)
try:
    shutil.rmtree(path)
except OSError as error:
    print(error)  

try:
    os.mkdir(path)
except OSError as error:
    print(error)  

#random.seed(0)
writer = SummaryWriter(comment="_VQVAE_TRNID_"+str(TRN_ID))
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
    model = VQVAE(vocabsize, dictmeta, ENC_NH, DEC_NH, Z_LEMMA_NH, Z_TAG_NH, DEC_DROPOUT, INPUT_EMB_DIM, NUM_CODEBOOK, NUM_CODEBOOK_ENTRIES, BIDIRECTIONAL)
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
    return model, opt, trnbatches, devbatches, tstbatches, trainset.charvocab, _trnbatches


def train(model, opt, trnbatches, devbatches, tstbatches, charvocab, _trnbatches):
    trnsize = sum([len(i[0]) for i in trnbatches.values()])
    devsize = sum([len(i[0]) for i in devbatches.values()])
    tstsize = sum([len(i[0]) for i in tstbatches.values()])

    logging.info("trnsize: %d" % trnsize)
    logging.info("devsize: %d" % devsize)
    logging.info("tstsize: %d" % tstsize)

    ## _trn
    """_trn_tgts   = defaultdict(lambda:0)
    _trn_tags   = defaultdict(lambda:0)
    _trn_lemmas = defaultdict(lambda:0)
    keys = list(_trnbatches.keys())
    for bid in keys:
        batch = _trnbatches[bid]
        tgt,tags,lemma = batch
        _trn_tgts['-'.join([str(sl) for sl in [l.item() for  l in tgt[0]]])] +=1
        _trn_lemmas['-'.join([str(sl) for sl in [l.item() for  l in lemma[0]]])] +=1
        _trn_tags['-'.join([str(sl) for sl in [l.item() for  l in tags[0]]])] +=1"""



    best_exact_match = 0
    most_used_trncode = 0
    most_used_devcode = 0

    kl_passed_epc = 0
    for epc in range(EPOCHS):
        logging.info("")
        logging.info("#epc: %d" % epc)

        epc_loss = dict({'trn':0, 'dev':0})
        epc_q_loss = dict({'trn':0, 'dev':0})
        epc_KL_loss = dict({'trn':0, 'dev':0})
        epc_q_indices = dict()
        epc_q_codes = dict()
        epc_nonpaddedtokens = dict({'trn':0, 'dev':0})
        epc_exactmatch = dict({'trn':0, 'dev':0})
        epc_tokensmatch = dict({'trn':0, 'dev':0})

        ## trn
        model.train()
        keys = list(trnbatches.keys())
        random.shuffle(keys)
        epc_q_indices['trn'] = dict()
        epc_q_indices['dev'] = dict()

        epc_q_codes['trn'] = defaultdict(lambda:0)
        epc_q_codes['dev'] = defaultdict(lambda:0)

        if epc >= KL_START_EPC:
            kl_passed_epc += 1
            epc_KL = min((kl_passed_epc/KL_ANNEAL_EPC)*KL_WEIGHT,KL_WEIGHT)
        else: 
            epc_KL = 0
        logging.info("epc:%d, epc_KL:  %.6f"%(epc,epc_KL))
        qjson_trn = dict()
        for bid in keys:
            batch = trnbatches[bid]
            tgt,tags = batch
            opt.zero_grad()
            batchloss, nonpadded_tokens, predtokens, Q, KL, quantized_indices, _, q_selections, quantized_codes = model(tgt, tags)

            for t in range(quantized_codes.shape[0]):
                code =   "-".join([str(i) for i in quantized_codes[t].tolist()])
                epc_q_codes['trn'][code] +=1
                word = ''.join(decode_batch(tgt[t].unsqueeze(1), charvocab))
                if code not in qjson_trn:
                    qjson_trn[code] = []
                qjson_trn[code].append(word)
            #optimize for per token nll + Q per word + KL per word
            optloss = (batchloss/nonpadded_tokens) +  (Q/tgt.size(0)) +  epc_KL * (KL/tgt.size(0))
            optloss.backward()
            opt.step()
            epc_loss['trn']+= batchloss.item()
            epc_nonpaddedtokens['trn'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, tgt[:,1:])
            epc_exactmatch['trn'] += exact_match
            epc_tokensmatch['trn'] += tokens_match
            epc_q_loss['trn'] += Q
            epc_KL_loss['trn'] += KL
            # keep number of different entry selections
            for idx,qi in enumerate(quantized_indices):
                qlist = qi.squeeze(1).tolist()
                for q in set(qlist):
                    if idx not in epc_q_indices['trn']:
                          epc_q_indices['trn'][idx] = defaultdict(lambda:0)
                    epc_q_indices['trn'][idx][q] +=qlist.count(q)
        #-end of trn
        with open(path+'/qinds_trn_epc'+str(epc)+'.json', 'w') as fp:
            json.dump(qjson_trn, fp, ensure_ascii=False, indent = 4)
     
        ## dev (teacher forcing)
        model.eval()
        for bid, batch in devbatches.items():
            tgt,tags = batch
            batchloss, nonpadded_tokens, predtokens, Q, KL, quantized_indices, _,q_selections, quantized_codes = model(tgt, tags)
            
            for t in range(quantized_codes.shape[0]):
                code =   "-".join([str(i) for i in quantized_codes[t].tolist()])
                epc_q_codes['dev'][code] +=1

            epc_loss['dev']+= batchloss.item()
            epc_nonpaddedtokens['dev'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, tgt[:,1:])
            epc_exactmatch['dev'] += exact_match
            epc_tokensmatch['dev'] += tokens_match
            epc_q_loss['dev'] += Q
            epc_KL_loss['dev'] += KL
            # keep number of different entry selections
            for idx,qi in enumerate(quantized_indices):
                qlist = qi.squeeze(1).tolist()
                for q in set(qlist):
                    if idx not in epc_q_indices['dev']:
                          epc_q_indices['dev'][idx] = defaultdict(lambda:0)
                    epc_q_indices['dev'][idx][q] +=qlist.count(q)


        ## Copy dev with one-step at a time
        out_df = pd.DataFrame({})
        exact_match_tst_acc=0
        i=0
        qjson = dict()
        qjsons = dict()
        qjsons[0] = dict()
        qjsons[1] = dict()
        #qjsons[2] = dict()

        for bid, batch in tstbatches.items():
            tgt,tags,lemma = batch
            predtokens, quantized_indices = model.decode(tgt,tags)
            QIDS = ';'.join([str(qi.item()) for qi in quantized_indices])
            lemmas = decode_batch(lemma[:,1:], charvocab)
            gold_decoded = decode_batch(tgt[:,1:], charvocab)
            pred_decoded = decode_batch(predtokens, charvocab)
            if QIDS not in qjson:
                qjson[QIDS] = []
            qjson[QIDS].append(gold_decoded[0])

            for di in range(len(QIDS.split(';'))):
                if QIDS.split(';')[di] not in qjsons[di]:
                    qjsons[di][QIDS.split(';')[di]] = []
                qjsons[di][QIDS.split(';')[di]].append(gold_decoded[0])

            """_predtokens = torch.cat((torch.tensor([2]).to('cuda'),predtokens[0]))
            predcount = '-'.join([str(sl) for sl in [l.item() for  l in _predtokens]])
            tgtcount = '-'.join([str(sl) for sl in [l.item() for  l in tgt[0]]])
            tagscount = '-'.join([str(sl) for sl in [l.item() for  l in tags[0]]])
            lcount = '-'.join([str(sl) for sl in [l.item() for  l in lemma[0]]])"""
            for g,p,l in zip(gold_decoded, pred_decoded, lemmas):
                if g == p:
                    out_df.at[i, "exact_match"] = 1
                    exact_match_tst_acc+=1
                else:
                    out_df.at[i, "exact_match"] = 0
                out_df.at[i, "lemma"] = l
                out_df.at[i, "gold"] = g
                out_df.at[i, "pred"] = p
                out_df.at[i, "selected_indices"] = ';'.join([str(qi.item()) for qi in quantized_indices])
                """out_df.at[i, "# pred seen in trn"]  = _trn_tgts[predcount]
                out_df.at[i, "# tgt seen in trn"]   = _trn_tgts[tgtcount]
                out_df.at[i, "# tags seen in trn"]  = _trn_tags[tagscount]
                out_df.at[i, "# lemma seen in trn"] = _trn_lemmas[lcount]"""
                i+=1
            if i>500:
                break
            out_df.to_csv(path+'/tst_copies_'+str(epc)+'.csv')
        tst_acc = exact_match_tst_acc / i
        logging.info("exact_match_tst_copy_acc: %.3f" % tst_acc)
        writer.add_scalar('exact_match_tst_copy_acc', tst_acc, epc)
        with open(path+'/qinds_tst_epc'+str(epc)+'.json', 'w') as fp:
            json.dump(qjson, fp, ensure_ascii=False, indent = 4)

        with open(path+'/qinds0_tst_epc'+str(epc)+'.json', 'w') as fp:
            json.dump(qjsons[0], fp, ensure_ascii=False, indent = 4)
        with open(path+'/qinds1_tst_epc'+str(epc)+'.json', 'w') as fp:
            json.dump(qjsons[1], fp, ensure_ascii=False, indent = 4)
        #with open(path+'/qinds2_tst_epc'+str(epc)+'.json', 'w') as fp:
        #    json.dump(qjsons[2], fp, ensure_ascii=False, indent = 4)

        ## Reinflect tst words - just expecting to see some reinflected lemma
        exact_match_tst_dictsupervised_wlemma = 0
        out_df = pd.DataFrame({})
        i=0
        for bid, batch in tstbatches.items():
            tgt,tags,lemma = batch
            predtokens, _ = model.decode(lemma,tags, mode='dict-supervised')
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
                i+=1
            if bid == 200:
                break
        out_df.to_csv(path+'/tst_reinflections_epc'+str(epc)+'.csv')
        
        ## Sample words with different lemmas
        out_df = pd.DataFrame({})
        entries =  [2,3,4,2,4,2,2,3,0,2,0]
        for i in range(50):
            predtokens = model.sample(entries)
            decoded_batches = decode_batch(predtokens, charvocab)
            out_df.at[i, "sampled"] = decoded_batches[0]
        out_df.to_csv(path+'/samples_epc'+str(epc)+'.csv')



        #logging.info('exact_match_tst_dictsupervised_wlemma: %.3f' % (exact_match_tst_dictsupervised_wlemma/i))
        #writer.add_scalar('reinflection_exact_match', (exact_match_tst_dictsupervised_wlemma/i), epc)



        ## log            
        ##loss per token
        logging.info("TRN || epcloss: %.4f" % (epc_loss['trn']/ epc_nonpaddedtokens['trn']))
        logging.info("    || Q: %.4f" % (epc_q_loss['trn']/trnsize))
        logging.info("    || KL: %.4f" % (epc_KL_loss['trn']/trnsize))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['trn']/epc_nonpaddedtokens['trn']), (epc_exactmatch['trn']/trnsize)))
        logging.info("    || unique taglist code: %d" %  len(epc_q_codes['trn']))
        # track number of different entry selections
        #for dictidx, qdict in epc_q_indices['trn'].items():
        #    logging.info("    || dict %d " % (dictidx))
        #    for entry_id, count in  dict(sorted(qdict.items())).items():
        #        logging.info("    || entry_id %d, ratio: %.3f" % (entry_id, count/sum(epc_q_indices['trn'][0].values())))
       

        logging.info("DEV || epcloss: %.4f" % (epc_loss['dev']/epc_nonpaddedtokens['dev']))
        logging.info("    || Q: %.4f" % (epc_q_loss['dev']/devsize))
        logging.info("    || KL: %.4f" % (epc_KL_loss['dev']/devsize))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['dev']/epc_nonpaddedtokens['dev']), (epc_exactmatch['dev']/devsize)))
        logging.info("    || unique taglist code: %d" %  len(epc_q_codes['dev']))
        # track number of different entry selections
        #for dictidx, qdict in epc_q_indices['dev'].items():
        #    logging.info("    || dict %d " % (dictidx))
        #    for entry_id, count in  dict(sorted(qdict.items())).items():
        #        logging.info("    || entry_id %d, ratio: %.3f" % (entry_id, count/sum(epc_q_indices['dev'][0].values())))
        seen_dev_code = 0
        for code, cnt in  epc_q_codes['dev'].items():
            if code in epc_q_codes['trn']:
                seen_dev_code+=1
        logging.info("    || seen taglist code: %d" %  seen_dev_code)

        writer.add_scalar('taglist_code/unique_trn_code', len(epc_q_codes['trn']), epc)
        writer.add_scalar('taglist_code/unique_dev_code', len(epc_q_codes['dev']), epc)
        writer.add_scalar('taglist_code/seen_dev_code', seen_dev_code, epc)

        if  len(epc_q_codes['trn']) > most_used_trncode:
            logging.info("    || MOST used trncode: %.d" % len(epc_q_codes['trn']))
            most_used_trncode = len(epc_q_codes['trn'])
            most_trn_epc = epc
        if  len(epc_q_codes['dev']) > most_used_devcode:
            logging.info("    || MOST used devcode: %.d" % len(epc_q_codes['dev']))
            most_used_devcode = len(epc_q_codes['dev'])
            most_dev_epc = epc

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
        writer.add_scalar('Q_loss/trn',   epc_q_loss['trn']/trnsize, epc)
        writer.add_scalar('Q_loss/dev',   epc_q_loss['dev']/devsize, epc)
        writer.add_scalar('KL_loss/trn',   epc_KL_loss['trn']/trnsize, epc)
        writer.add_scalar('KL_loss/dev',   epc_KL_loss['dev']/devsize, epc)
        writer.add_scalar('token_match/trn',epc_tokensmatch['trn']/ epc_nonpaddedtokens['trn'], epc)
        writer.add_scalar('token_match/dev',  epc_tokensmatch['dev']/ epc_nonpaddedtokens['dev'], epc)
        writer.add_scalar('exact_match/trn',epc_exactmatch['trn']/trnsize, epc)
        writer.add_scalar('exact_match/dev',  epc_exactmatch['dev']/devsize, epc)
    logging.info("Training is over, saved model with best dev exact match: %.3f (epc %d)" % (best_exact_match/devsize, best_epc))
    logging.info("Training is over, most used trncode at %d, most used devcode at %d " % (most_trn_epc, most_dev_epc))


model, opt, trn, dev, tst, charvocab, _trnbatches = config()
train(model, opt, trn, dev, tst, charvocab, _trnbatches)