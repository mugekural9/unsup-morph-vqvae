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

BATCHSIZE = 64
EPOCHS = 200
LR = 0.0005
KL_WEIGHT = 0.1
KL_ANNEAL_EPC = 10
KL_START_EPC = 5
ENC_NH = 256
DEC_NH = 256
Z_NH = 128
DEC_DROPOUT = 0.2
INPUT_EMB_DIM = 64
TRN_ID = "tur_mergedstdata_VERBS_128k_z"+str(Z_NH)+"_dec_nh"+str(DEC_NH)+"_kl"+str(KL_WEIGHT)+"_epc"+str(KL_ANNEAL_EPC)+"_start"+str(KL_START_EPC)+"_dec_dropout"+str(DEC_DROPOUT)+"_input_emb_dim"+str(INPUT_EMB_DIM)+"_batchsize"+str(BATCHSIZE)
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

writer = SummaryWriter(comment="_VQVAE_TRNID_"+str(TRN_ID))
device = "cuda" if torch.cuda.is_available() else "cpu"


def config():
    logging.basicConfig(handlers=[
            logging.FileHandler("results/vqvae/"+str(TRN_ID)+"/training_vqvae.log"),
            logging.StreamHandler()],
            format='%(asctime)s - %(message)s', level=logging.INFO)
    trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata_VERBS")
    #trainset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur_large.train")
    devset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.dev_VERBS", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)
    tstset = MorphDataset("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold_VERBS", charvocab=trainset.charvocab, tagsvocab=trainset.tagsvocab)

    maxtrnsize = 128000
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
        trn_tagsets.append(set(taglist.split(';')))
        trn_tagsets_dict[str(set(taglist.split(';')))]+=1

    for taglist in devset.tagslist:
        if set(taglist.split(';')) in trn_tagsets:
            seen_dev_taglist +=1
        dev_tagsets_dict[str(set(taglist.split(';')))] += 1

    for taglist in tstset.tagslist:
        if set(taglist.split(';')) in trn_tagsets:
            seen_tst_taglist +=1
        tst_tagsets_dict[str(set(taglist.split(';')))] += 1


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
    model = VQVAE(vocabsize,dictmeta, ENC_NH, DEC_NH, Z_NH, DEC_DROPOUT, INPUT_EMB_DIM)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=LR,  betas=(0.5, 0.999))
    logging.info("trnID: %s" % str(TRN_ID))
    logging.info("batchsize: %d" % BATCHSIZE)
    logging.info("epochs: %d" % EPOCHS)
    logging.info("lr: %.5f" % LR)
    logging.info("KL_WEIGHT: %.5f" % KL_WEIGHT)
    logging.info("opt: %s", opt)

    logging.info(model)
    with open("results/vqvae/"+str(TRN_ID)+"/charvocab.json", "w") as outfile:
        json.dump(trainset.charvocab.id2char, outfile, ensure_ascii=False, indent=4)

    for tag,vocab in trainset.tagsvocab.vocabs.items():
        with open("results/vqvae/"+str(TRN_ID)+"/"+tag+"_vocab.json", "w") as outfile:
            json.dump(vocab.id2tag, outfile, ensure_ascii=False, indent=4)
    return model, opt, trnbatches, devbatches, tstbatches, trainset.charvocab, _trnbatches


def train(model, opt, trnbatches, devbatches, tstbatches, charvocab, _trnbatches):
    trnsize = sum([len(i[0]) for i in trnbatches.values()])
    devsize = sum([len(i[0]) for i in devbatches.values()])
    tstsize = sum([len(i[0]) for i in tstbatches.values()])

    logging.info("trnsize: %d" % trnsize)
    logging.info("devsize: %d" % devsize)
    
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
    for epc in range(EPOCHS):
        logging.info("")
        logging.info("#epc: %d" % epc)

        epc_loss = dict({'trn':0, 'dev':0})
        epc_q_loss = dict({'trn':0, 'dev':0})
        epc_KL_loss = dict({'trn':0, 'dev':0})
        epc_q_indices = dict()
        epc_nonpaddedtokens = dict({'trn':0, 'dev':0})
        epc_exactmatch = dict({'trn':0, 'dev':0})
        epc_tokensmatch = dict({'trn':0, 'dev':0})

        ## trn
        model.train()
        keys = list(trnbatches.keys())
        random.shuffle(keys)
        epc_q_indices['trn'] = dict()
        epc_q_indices['dev'] = dict()

        #epc_q_selections = dict()
        #epc_q_selections['trn'] = dict()
        #epc_q_selections['dev'] = dict()
        #for dictidx in range(len(model.quantizers)):
        #    epc_q_selections['trn'][dictidx] = dict()
        #    epc_q_selections['trn'][dictidx]['true'] = 0
        #    epc_q_selections['trn'][dictidx]['total'] = 0
        #    epc_q_selections['dev'][dictidx] = dict()
        #    epc_q_selections['dev'][dictidx]['true'] = 0
        #    epc_q_selections['dev'][dictidx]['total'] = 0

        if epc >= KL_START_EPC:
            epc_KL = min((epc/KL_ANNEAL_EPC)*KL_WEIGHT,KL_WEIGHT)
        else: 
            epc_KL = 0
        logging.info("epc:%d, epc_KL:  %.6f"%(epc,epc_KL))
        for bid in keys:
            batch = trnbatches[bid]
            tgt,tags = batch
            opt.zero_grad()
            batchloss, nonpadded_tokens, predtokens, Q, KL, quantized_indices, _, q_selections = model(tgt, tags)
            
            #track dict selections
            #for dictidx, selections in q_selections.items():
            #    true_selections, total_selections = selections
            #    epc_q_selections['trn'][dictidx]['true']  += true_selections 
            #    epc_q_selections['trn'][dictidx]['total'] += total_selections 

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
            for idx,qi in enumerate(quantized_indices):
                qlist = qi.squeeze(1).tolist()
                for q in set(qlist):
                    if idx not in epc_q_indices['trn']:
                          epc_q_indices['trn'][idx] = defaultdict(lambda:0)
                    epc_q_indices['trn'][idx][q] +=qlist.count(q)
            
        ## dev (teacher forcing)
        model.eval()
        for bid, batch in devbatches.items():
            tgt,tags = batch
            batchloss, nonpadded_tokens, predtokens, Q, KL, quantized_indices, _,q_selections = model(tgt, tags)
            
            #track dict selections     
            #for dictidx, selections in q_selections.items():
            #    true_selections, total_selections = selections
            #    epc_q_selections['dev'][dictidx]['true']  += true_selections 
            #    epc_q_selections['dev'][dictidx]['total'] += total_selections 

            epc_loss['dev']+= batchloss.item()
            epc_nonpaddedtokens['dev'] += nonpadded_tokens
            tokens_match, _, exact_match = accuracy_on_batch(predtokens, tgt[:,1:])
            epc_exactmatch['dev'] += exact_match
            epc_tokensmatch['dev'] += tokens_match
            epc_q_loss['dev'] += Q
            epc_KL_loss['dev'] += KL

        out_df = pd.DataFrame({})
        exact_match_dev_dictsupervised_wtgt=0
        if epc>30:
            i=0
            for bid, batch in tstbatches.items():
                tgt,tags,lemma = batch
                predtokens = model.decode(tgt,tags)
                lemmas = decode_batch(lemma[:,1:], charvocab)
                gold_decoded_batches = decode_batch(tgt[:,1:], charvocab)
                pred_decoded_batches = decode_batch(predtokens, charvocab)
                
                """_predtokens = torch.cat((torch.tensor([2]).to('cuda'),predtokens[0]))
                predcount = '-'.join([str(sl) for sl in [l.item() for  l in _predtokens]])
                tgtcount = '-'.join([str(sl) for sl in [l.item() for  l in tgt[0]]])
                tagscount = '-'.join([str(sl) for sl in [l.item() for  l in tags[0]]])
                lcount = '-'.join([str(sl) for sl in [l.item() for  l in lemma[0]]])"""
                for g,p,l in zip(gold_decoded_batches, pred_decoded_batches, lemmas):
                    centry = ""
                    for k in range(len(tags[0])):
                        centry += str(tags[0][k].item()) + "-"
                    out_df.at[i, "FULL_codebook_entry"] = centry
                    if g == p:
                        out_df.at[i, "exact_match"] = 1
                        exact_match_dev_dictsupervised_wtgt+=1
                    else:
                        out_df.at[i, "exact_match"] = 0
                    out_df.at[i, "lemma"] = l
                    out_df.at[i, "gold"] = g
                    out_df.at[i, "pred"] = p
                    """out_df.at[i, "# pred seen in trn"]  = _trn_tgts[predcount]
                    out_df.at[i, "# tgt seen in trn"]   = _trn_tgts[tgtcount]
                    out_df.at[i, "# tags seen in trn"]  = _trn_tags[tagscount]
                    out_df.at[i, "# lemma seen in trn"] = _trn_lemmas[lcount]"""
                    i+=1
                if i>100:
                    break
            out_df.to_csv(path+'/dev_wtgts_epc'+str(epc)+'.csv')
            logging.info("exact_match_dev_dictsupervised_wtgt: %d" % exact_match_dev_dictsupervised_wtgt)

        out_df = pd.DataFrame({})
        verb_entries =  [2,3,4,2,4,2,2,3,0,2,0]
        for i in range(50):
            predtokens = model.sample(verb_entries)
            decoded_batches = decode_batch(predtokens, charvocab)
            out_df.at[i, "sampled"] = decoded_batches[0]
        out_df.to_csv(path+'/samples_epc'+str(epc)+'.csv')


        """
        exact_match_tst_dictunsupervised_wtgt = 0
        exact_match_tst_dictsupervised_wtgt = 0

        ## tst (decoding for copying dict-unsupervised)
        out_df = pd.DataFrame({})
        i=0
        for bid, batch in tstbatches.items():
            tgt,tags,lemma = batch
            _,_, _, _, _, quantized_indices, _ = model(tgt, tags, mode='dict-unsupervised')
            quantized_indices = torch.tensor([qi.item() for qi in quantized_indices]).unsqueeze(0).to(device)
            predtokens = model.decode(tgt,quantized_indices)
            lemmas = decode_batch(lemma[:,1:], charvocab)
            gold_decoded_batches = decode_batch(tgt[:,1:], charvocab)
            pred_decoded_batches = decode_batch(predtokens, charvocab)
            for g,p,l in zip(gold_decoded_batches, pred_decoded_batches, lemmas):
                out_df.at[i, "gold"] = g
                out_df.at[i, "pred"] = p
                out_df.at[i, "lemma"] = l
                centry = ""
                for k in range(len(quantized_indices[0])):
                    centry += str(quantized_indices[0][k].item()) + "-"
                out_df.at[i, "FULL_codebook_entry"] = centry
                if g == p:
                    out_df.at[i, "exact_match"] = 1
                    exact_match_tst_dictunsupervised_wtgt+=1
                else:
                    out_df.at[i, "exact_match"] = 0
                i+=1
            #if bid == 49:
            #    break
        out_df.to_csv(path+'/unsup_wtgts.csv')
        
        ## tst (decoding for copying dict-supervised with tgt)
        out_df = pd.DataFrame({})
        i=0
        for bid, batch in tstbatches.items():
            tgt,tags,lemma = batch
            predtokens = model.decode(tgt,tags)
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
                    exact_match_tst_dictsupervised_wtgt+=1
                else:
                    out_df.at[i, "exact_match"] = 0
                i+=1
            if bid == 49:
                break
        out_df.to_csv(path+'/sup_wtgts.csv')
       
       
        """
        ## tst (decoding for for copying dict-supervised with lemma)
        exact_match_tst_dictsupervised_wlemma = 0
        out_df = pd.DataFrame({})
        i=0
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
            if bid == 300:
                break
        out_df.to_csv(path+'/sup_wlemmas.csv')
        logging.info('exact_match_tst_dictsupervised_wlemma: %.3f' % (exact_match_tst_dictsupervised_wlemma/i))
        writer.add_scalar('reinflection_exact_match', (exact_match_tst_dictsupervised_wlemma/i), epc)

        ## log            
        ##loss per token
        logging.info("TRN || epcloss: %.4f" % (epc_loss['trn']/ epc_nonpaddedtokens['trn']))
        logging.info("    || Q: %.4f" % (epc_q_loss['trn']/trnsize))
        logging.info("    || KL: %.4f" % (epc_KL_loss['trn']/trnsize))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['trn']/epc_nonpaddedtokens['trn']), (epc_exactmatch['trn']/trnsize)))
        
        """_true_trn = 0
        _total_trn = 0
        for idx, selections in epc_q_selections['trn'].items():
            _true  = selections['true']
            _total = selections['total']
            _acc = _true/_total
            logging.info("    || dict %d, selection acc: %.3f" % (idx, _acc))
            writer.add_scalar('dict_selection/trn/'+str(idx), _acc, epc)
            _true_trn   +=  _true
            _total_trn  +=  _total
        logging.info("    || avg_dict_selection: %.3f"  % ( _true_trn/_total_trn))
        writer.add_scalar('dict_selection_avg/trn', _true_trn/_total_trn, epc)"""

        logging.info("DEV || epcloss: %.4f" % (epc_loss['dev']/epc_nonpaddedtokens['dev']))
        logging.info("    || Q: %.4f" % (epc_q_loss['dev']/devsize))
        logging.info("    || KL: %.4f" % (epc_KL_loss['dev']/devsize))
        logging.info("    || token_match: %.3f, exact_match: %.3f" % ((epc_tokensmatch['dev']/epc_nonpaddedtokens['dev']), (epc_exactmatch['dev']/devsize)))
        
        """_true_dev = 0
        _total_dev = 0
        for idx, selections in epc_q_selections['dev'].items():
            _true  = selections['true']
            _total = selections['total']
            _acc = _true/_total
            logging.info("    || dict %d, selection acc: %.3f" % (idx, _acc))
            writer.add_scalar('dict_selection/dev/'+str(idx), _acc, epc)
            _true_dev   +=  _true
            _total_dev  +=  _total
        logging.info("    || avg_dict_selection: %.3f"  % ( _true_dev/_total_dev))
        writer.add_scalar('dict_selection_avg/dev', _true_dev/_total_dev, epc)"""
        
        
 

        """logging.info("TST ||   ")
        logging.info("    || exact match with exact_match_tst_dictunsupervised_wtgt: %.3f" % (exact_match_tst_dictunsupervised_wtgt/50))
        logging.info("    || exact match with exact_match_tst_dictsupervised_wtgt: %.3f" % (exact_match_tst_dictsupervised_wtgt/50))
        logging.info("    || exact match with exact_match_tst_dictsupervised_wlemma: %.3f" % (exact_match_tst_dictsupervised_wlemma/50))"""
    
        ## save
        if  epc_exactmatch['dev'] > best_exact_match:
            logging.info("    || BEST exact match (teacher forcing): %.3f" % (epc_exactmatch['dev']/devsize))
            best_exact_match = epc_exactmatch['dev']
            best_epc = epc
        
        if epc>30 and epc%3==0:
            torch.save(model.state_dict(), "results/vqvae/"+str(TRN_ID)+"/vqvae_"+str(epc)+".pt")
            logging.info("    || saved model")

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


model, opt, trn, dev, tst, charvocab, _trnbatches = config()
train(model, opt, trn, dev, tst, charvocab, _trnbatches)