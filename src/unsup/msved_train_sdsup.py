# -----------------------------------------------------------
# Date:        2021/12/20 
# Author:      Muge Kural
# Description: Trainer of character-based variational-autoencoder model, saves the results under ./results directory.
# -----------------------------------------------------------

import sys, argparse, random, torch, json, matplotlib, os, math
import matplotlib.pyplot as plt
import numpy as np
from model.msved.msved_sdsup import MSVED
from common.utils import *
from torch import optim
from data.data import build_data
matplotlib.use('Agg')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   

def test(batches, mode, args, kl_weight, tmp):
    numwords = args.valsize if mode =='val'  else args.tstsize
    indices = list(range( len(batches)))
    epoch_loss = 0
    epoch_recon_loss = 0
    epoch_tag_total_tokens = 0; epoch_tag_correct= 0; 
    epoch_labeled_pred_loss = 0; epoch_labeled_recon_loss = 0
    epoch_labeled_num_tokens = 0
    epoch_labeled_kl_loss = 0
    epoch_labeled_reinflect_recon_acc = 0
    for i, idx in enumerate(indices):
        # (batchsize)
        surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, reinflect_surf  = batches[idx] 
        # (batchsize)
        loss, labeled_pred_loss, tag_correct, tag_total, labeled_recon_loss, labeled_kl_loss, labeled_reinflect_recon_acc  = args.model.loss_l(surf,case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, reinflect_surf, kl_weight, tmp, mode='test')
        epoch_tag_correct += tag_correct
        epoch_tag_total_tokens += tag_total
        epoch_labeled_num_tokens +=  torch.sum(reinflect_surf[:,1:] !=0).item()
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
    args.logger.write('%s--- loss: %.4f, labeled_pred_loss: %.4f, labeled_pred_acc: %.4f, labeled_recon_loss: %.4f,  labeled_kl_loss: %.4f,  labeled_reinflect_recon_acc: %.4f \n' % (mode, loss, labeled_pred_loss, labeled_pred_acc, labeled_recon_loss,  labeled_kl_loss, labeled_reinflect_recon_acc))
    return loss, recon

def train(data, args):
    trnbatches, valbatches, tstbatches = data
    # initialize optimizer
    opt = optim.Adam(filter(lambda p: p.requires_grad, args.model.parameters()), lr=args.lr)
    # Log trainable model parameters
    for name, prm in args.model.named_parameters():
        args.logger.write('\n'+name+', '+str(prm.shape) + ': '+ str(prm.requires_grad))
    numbatches = len(trnbatches); indices = list(range(numbatches))
    numwords = args.trnsize
    best_loss = 1e4
    tmp=1.0
    update_ind =0
    for epc in range(args.epochs):
        epoch_loss = 0; epoch_num_tokens = 0; 
        epoch_tag_total_tokens = 0; epoch_tag_correct= 0; 
        epoch_labeled_pred_loss = 0; epoch_labeled_recon_loss = 0; 
        epoch_labeled_kl_loss = 0; 
        epoch_labeled_num_tokens = 0
        epoch_labeled_reinflect_recon_acc = 0
        random.shuffle(indices) # this breaks continuity if there is any

        for i, lidx in enumerate(indices):
            loss = torch.tensor(0.0).to('cuda')
            if update_ind % args.update_temp == 0:
                tmp = get_temp(update_ind)
            kl_weight = get_kl_weight(update_ind, 0.2, 150000.0)
            args.model.zero_grad()
            update_ind +=1
            lidx= indices[i]
            lx_src, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lx_tgt  = trnbatches[lidx] 
            # (batchsize)
            loss_l, labeled_pred_loss, tag_correct, tag_total, labeled_recon_loss,  labeled_kl_loss, labeled_reinflect_recon_acc = args.model.loss_l(lx_src,case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, lx_tgt, kl_weight, tmp)
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
         
        loss = epoch_loss / numwords  
        labeled_pred_loss = epoch_labeled_pred_loss/ epoch_tag_total_tokens
        labeled_pred_acc  = epoch_tag_correct/ epoch_tag_total_tokens
        labeled_recon_loss = epoch_labeled_recon_loss / epoch_labeled_num_tokens
        labeled_kl_loss = epoch_labeled_kl_loss / numwords
        labeled_reinflect_recon_acc = epoch_labeled_reinflect_recon_acc / epoch_labeled_num_tokens

        args.logger.write('\nepoch: %.1d, kl_weight: %.2f, tmp: %.2f' % (epc, kl_weight, tmp))
        args.logger.write('\ntrn--- loss: %.4f, labeled_pred_loss: %.4f, labeled_pred_acc: %.4f, labeled_recon_loss: %.4f,  labeled_kl_loss: %.4f, labeled_reinflect_recon_acc: %.4f \n' % (loss, labeled_pred_loss, labeled_pred_acc, labeled_recon_loss,  labeled_kl_loss,  labeled_reinflect_recon_acc))
        # VAL
        args.model.eval()
        with torch.no_grad():
            loss, recon = test(valbatches, "val", args, kl_weight, tmp)
        if loss < best_loss:
            args.logger.write('update best loss \n')
            best_loss = loss
            torch.save(args.model.state_dict(), args.save_path)
        # SHARED TASK
        if epc % 10 == 0:
            shared_task_gen(tstbatches, args)
        args.model.train()

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

def shared_task_gen(batches, args):
    indices = list(range( len(batches)))
    correct = 0
    with open('shared_task_tst_beam_sdsup.txt', 'w') as f:
        for i, idx in enumerate(indices):
            # (batchsize)
            surf, case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss, gold_reinflect_surf  = batches[idx] 
            inflected_form = ''.join(surf_vocab.decode_sentence(surf.squeeze(0)))
            reinflected_form = args.model.generate(surf,case,polar,mood,evid,pos,per,num,tense,aspect,inter,poss)
            gold_reinflected_form = ''.join(surf_vocab.decode_sentence(gold_reinflect_surf.squeeze(0)))
            f.write(inflected_form+'\t'+reinflected_form+ '\t'+gold_reinflected_form+ '\n')
            #gold_reinflected_form = gold_reinflected_form[3:]
            if reinflected_form == gold_reinflected_form:
             correct +=1
    args.logger.write('TST SET ACCURACY: %.3f' % (correct/1600))
    

# CONFIG
parser = argparse.ArgumentParser(description='')
args = parser.parse_args()
args.device = 'cuda'
# training
args.batchsize = 128; args.epochs = 1
args.opt= 'Adam'; args.lr = 0.001
args.task = 'msved'
args.seq_to_no_pad = 'surface'
# data
args.trndata  = 'data/sigmorphon2016/turkish-task3-train'
args.valdata  = 'data/sigmorphon2016/turkish-task3-test'
args.tstdata  = 'data/sigmorphon2016/turkish-task3-test'

args.update_temp = 2000
args.surface_vocab_file = args.trndata
args.maxtrnsize = 10000000; args.maxvalsize = 10000; args.maxtstsize = 10000
rawdata, batches, surf_vocab, tag_vocabs = build_data(args)
for key,val in tag_vocabs.items():
    with open(key+'_vocab.json', 'w') as f:
        f.write(json.dumps(val))
breakpoint()

trndata, vlddata, tstdata = rawdata
args.trnsize , args.valsize, args.tstsize = len(trndata), len(vlddata), len(trndata)
# model
args.mname = 'msved' 
model_init = uniform_initializer(0.01)
emb_init = uniform_initializer(0.1)
args.ni = 300; args.nz = 150; 
args.enc_nh = 256; args.dec_nh = 256
args.enc_dropout_in = 0.0; args.enc_dropout_out = 0.0
args.dec_dropout_in = 0.5; 
args.model = MSVED(args, surf_vocab, tag_vocabs, model_init, emb_init)
args.model.to(args.device)

# logging
args.modelname = 'model/'+args.mname+'/results/training/'+str(len(trndata))+'_instances/'
try:
    os.makedirs(args.modelname)
    print("Directory " , args.modelname ,  " Created ") 
except FileExistsError:
    print("Directory " , args.modelname ,  " already exists")
args.save_path = args.modelname +  str(args.epochs)+'epochs.pt'
args.log_path =  args.modelname +  str(args.epochs)+'epochs.log'
args.fig_path =  args.modelname +  str(args.epochs)+'epochs.png'
args.logger = Logger(args.log_path)
with open(args.modelname+'/surf_vocab.json', 'w') as f:
    f.write(json.dumps(surf_vocab.word2id))
args.logger.write('\nnumber of params: %d \n' % count_parameters(args.model))
args.logger.write(args)
args.logger.write('\n')
# plotting
args.fig, args.axs = plt.subplots(3)
args.plt_style = pstyle = '-'
args.fig.tight_layout() 

# RUN
train(batches, args)
plt.savefig(args.fig_path)