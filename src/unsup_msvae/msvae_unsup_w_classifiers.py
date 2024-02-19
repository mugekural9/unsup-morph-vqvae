# -----------------------------------------------------------
# Date:        2021/12/19 
# Author:      Muge Kural
# Description: Character-based Variational Autoencoder 
# -----------------------------------------------------------

import math
from pprint import pprint
from tokenize import Ignore
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class MSVED_Encoder(nn.Module):
    """ LSTM Encoder with constant-length batching"""
    def __init__(self, args, embed, model_init, emb_init, bidirectional=True):
        super(MSVED_Encoder, self).__init__()
        self.ni = args.ni
        self.nh = args.enc_nh
        self.nz = args.nz
        self.embed = embed

        self.gru = nn.GRU(input_size=args.ni,
                            hidden_size=args.enc_nh,
                            num_layers=1,
                            batch_first=True,
                            dropout=0,
                            bidirectional=bidirectional)

        self.dropout_in = nn.Dropout(args.enc_dropout_in)

        # dimension transformation to z
        if self.gru.bidirectional:
            self.linear = nn.Linear(args.enc_nh*2, 2*args.nz, bias=True)
        else:
            self.linear = nn.Linear(args.enc_nh,  2*args.nz, bias=False)

        self.reset_parameters(model_init, emb_init)
        #nn.init.xavier_normal_(self.linear.weight)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.embed.weight)

    def forward(self, input):
        # (batch_size, seq_len-1, args.ni)
        word_embed = self.embed(input)
        word_embed = self.dropout_in(word_embed)

        _, last_state = self.gru(word_embed)
        if self.gru.bidirectional:
            last_state = torch.cat([last_state[-2], last_state[-1]], 1).unsqueeze(0)
        mean, logvar = self.linear(last_state).chunk(2, -1)
        # (batch_size, 1, enc_nh)
        last_state = last_state.permute(1,0,2)
        return mean.squeeze(0), logvar.squeeze(0), last_state

class MSVED_Decoder(nn.Module):
    """LSTM decoder with constant-length batching"""
    def __init__(self, args, embed, charvocab, model_init, emb_init, NUM_CODEBOOK, NUM_CODEBOOK_ENTRIES, tag_embed_dim):
        super(MSVED_Decoder, self).__init__()
        self.ni = args.ni
        self.nh = args.dec_nh
        self.nz = args.nz
        self.vocab = charvocab
        self.device = args.device
        self.char_embed = embed
        self.tag_embed_dim = tag_embed_dim

        # no padding when setting padding_idx to -1
        #self.char_embed = nn.Embedding(len(vocab.word2id), 300, padding_idx=0)

        self.dropout_in = nn.Dropout(args.dec_dropout_in)

        # concatenate z with input
        self.gru = nn.GRU(input_size=self.ni+ self.nz+self.tag_embed_dim, # self.char_embed+ self.ni
                            hidden_size=self.nh,
                            num_layers=1,
                            batch_first=True)

        self.attn = nn.Linear(self.ni+ self.nz+ self.nh, NUM_CODEBOOK)
        self.attn_combine = nn.Linear(self.ni+ self.nz+self.tag_embed_dim, self.ni+ self.nz+self.tag_embed_dim)

        # prediction layer
        self.pred_linear = nn.Linear(args.dec_nh, len(charvocab.char2id), bias=True)
        vocab_mask = torch.ones(len(charvocab.char2id))
        self.loss = nn.CrossEntropyLoss(weight=vocab_mask, reduce=False, ignore_index=0)
        self.reset_parameters(model_init, emb_init)

    def reset_parameters(self, model_init, emb_init):
        for param in self.parameters():
            model_init(param)
        emb_init(self.char_embed.weight)
        #    torch.nn.init.xavier_normal_(param)

    def forward(self, input, z, hidden, tag_embeddings):
        # input: (batch_size,1), hidden(1,batch_size,hout)
        batch_size, _, _ = z.size()
        seq_len = input.size(1)
        # (batch_size, seq_len, ni)
        embedded = self.char_embed(input)
        embedded = self.dropout_in(embedded)
        z_ = z.expand(batch_size, seq_len, self.nz)

        # (batch_size, 1, ni+nz)
        embedded = torch.cat((embedded, z_), -1)

        # (batchsize,1, ni+nz+hiddensize)
        # Attention Queries
        to_attend = torch.cat((embedded, torch.permute(hidden, (1,0,2))), 2) 

        # Attention Keys: self.attn
        # (batchsize, 1, 11)
        attention_scores = self.attn(to_attend)
        # NO MASKING! Because we don't know the tags.
        #attention_scores = attention_scores.masked_fill(tag_attention_masks, -1e9)
        
        attn_weights = F.softmax(attention_scores, dim=2)
        # (batchsize, 1, self.tag_embed_dim)
        attn_applied = torch.bmm(attn_weights,tag_embeddings)
        # (batchsize,1, ni+nz+tagsize)
        output = torch.cat((embedded, attn_applied), 2)


        output = self.attn_combine(output)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)
        # (batch_size, 1, vocab_size)
        output_logits = self.pred_linear(output)
        return output_logits, hidden, attn_weights

class MSVAE(nn.Module):
    def __init__(self, args, charvocab, dictmeta, model_init, emb_init, NUM_CODEBOOK, NUM_CODEBOOK_ENTRIES, TAG_EMBED_DIM, IS_HARD_TAU):
        super(MSVAE, self).__init__()
        self.embed = nn.Embedding(len(charvocab.char2id), args.ni)
       
        self.args = args
        self.nz = args.nz
        self.tag_embed_dim = TAG_EMBED_DIM
        self.dec_nh = args.dec_nh
        self.char_emb_dim = args.ni
        self.z_to_dec = nn.Linear(self.nz, self.dec_nh)
        self.tag_to_dec = nn.Linear(self.tag_embed_dim, self.dec_nh)
        self.is_hard_tau = IS_HARD_TAU
        self.encoder = MSVED_Encoder(args, self.embed, model_init, emb_init)
        self.decoder = MSVED_Decoder(args, self.embed, charvocab, model_init, emb_init, NUM_CODEBOOK, NUM_CODEBOOK_ENTRIES, self.tag_embed_dim)

        #torch.nn.init.xavier_uniform(self.z_to_dec.weight)
        #torch.nn.init.xavier_uniform(self.tag_to_dec.weight)

        loc = torch.zeros(self.nz, device=args.device)
        scale = torch.ones(self.nz, device=args.device)
        self.prior = torch.distributions.normal.Normal(loc, scale)
        self.tag_embeddings = nn.ModuleList([])
        self.classifiers = nn.ModuleList([])
        self.tag_embeddings_biases = []
        self.priors = []
        self.NUM_CODEBOOK = NUM_CODEBOOK
        # Discriminative classifiers for q(y|x)
        for keydict in dictmeta:
            self.classifiers.append(nn.Linear(256*2, keydict))
            #    nn.init.xavier_normal_(self.classifiers[-1].weight)
            self.tag_embeddings.append(nn.Embedding(keydict, self.tag_embed_dim))
            self.priors.append(torch.zeros(1,keydict))
            self.tag_embeddings_biases.append(nn.Parameter(torch.ones(1,self.tag_embed_dim)).to('cuda'))

    def classifier_loss(self, enc_nh, tmp):
        sft = nn.Softmax(dim=2)
        preds =[]
        xloss = torch.tensor(0.0).to('cuda')
        gumbel_tag_embeddings = []
        for i in range(len(self.classifiers)):
            # (batchsize,1,tagvocabsize)
            logits = self.classifiers[i](enc_nh)
            logits = torch.tanh(logits)
            preds.append(torch.argmax(sft(logits),dim=2))
            # (batchsize,tagvocabsize)
            gumbel_logits = F.gumbel_softmax(logits, tau=tmp, hard=self.is_hard_tau).squeeze(1)
            gumbel_tag_embeddings.append(torch.matmul(gumbel_logits, self.tag_embeddings[i].weight).unsqueeze(1))
        
        # (batchsize, NUMCODEBOOK)
        preds =  torch.stack(preds,dim=1).squeeze(2)
        tag_correct = 0; tag_total = 0
        # (batchsize, 11, self.tag_embed_dim)
        gumbel_tag_embeddings = torch.cat(gumbel_tag_embeddings, dim=1)
        return gumbel_tag_embeddings, xloss, tag_correct, tag_total, preds

    def loss_l(self, lx_src, tags, lx_tgt, kl_weight, tmp, mode='train'):
        # [Ll (xt, yt| xs) - D(xt|yt)]
        labeled_msved_loss, labeled_pred_loss, tag_correct, tag_total, labeled_recon_loss, labeled_kl_loss, labeled_recon_acc, preds, recon_preds = self.labeled_msved_loss(lx_src, tags, lx_tgt, kl_weight, tmp, mode=mode)
        loss = labeled_msved_loss
        return loss, labeled_pred_loss, tag_correct, tag_total, labeled_recon_loss, labeled_kl_loss, labeled_recon_acc, preds, recon_preds

    def labeled_msved_loss(self, x, tags, reinflect_surf, kl_weight, tmp, mode='train'):
        # Ll (xt, yt | xs)
        mu, logvar, encoder_fhs = self.encoder(x)
        gumbel_tag_embeddings, xloss, tag_correct, tag_total, preds = self.classifier_loss(encoder_fhs, tmp)
        if mode == 'train':
            # (batchsize, 1, nz)
            z = self.reparameterize(mu, logvar)
        else:
            z = mu.unsqueeze(1)
        
        tag_embeds = []
        for i in range(gumbel_tag_embeddings.shape[1]):
            #(batchsize, self.tag_embed_dim)
            tag_emb = gumbel_tag_embeddings[:,i,:]
            tag_embeds.append(tag_emb)
        # (batchsize, 11, self.tag_embed_dim)
        tag_embeddings = torch.stack(tag_embeds,dim=1)
        
        # (batchsize, 1, self.tag_embed_dim)
        tag_all_embed = torch.sum(tag_embeddings,dim=1).unsqueeze(1)
        #TODO: add bias
        tag_all_embed = torch.tanh(tag_all_embed)
        dec_h0 = torch.tanh(self.tag_to_dec(tag_all_embed) + self.z_to_dec(z))
        #(1,batchsize, self.dec_nh)
        dec_h0 = torch.permute(dec_h0, (1,0,2))

        #workaround to have tst preds
        recon_preds= None
        if mode == 'train':
            recon_loss, recon_acc = self.recon_loss(reinflect_surf, z, dec_h0, tag_embeddings, recon_type='sum')
        else:
            recon_loss, recon_acc, recon_preds = self.recon_loss_test(reinflect_surf, z, dec_h0, tag_embeddings, recon_type='sum')
        
        
        # (batchsize)
        kl_loss = self.kl_loss(mu,logvar)
        # (batchsize)
        recon_loss = recon_loss.squeeze(1)#.mean()
        loss =  recon_loss + kl_weight * kl_loss
        return loss, xloss, tag_correct, tag_total, recon_loss, kl_loss, recon_acc, preds, recon_preds

    def kl_loss(self, mu, logvar):
        # KL: (batch_size), mu: (batch_size, nz), logvar: (batch_size, nz)
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        return KL

    def recon_loss(self, y, z, decoder_hidden, tag_attention_values, recon_type='avg'):
        #remove end symbol
        src = y[:, :-1]
        # remove start symbol
        tgt = y[:, 1:]        
        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        output_logits = []
        for di in range(seq_len):
            decoder_input = src[:,di].unsqueeze(1)  # Teacher forcing
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, z, decoder_hidden, tag_attention_values)
            output_logits.append(decoder_output)
        
        # (batchsize, seq_len, vocabsize)
        output_logits = torch.cat(output_logits,dim=1)

        _tgt = tgt.contiguous().view(-1)
        
        # (batch_size  * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))

        # (batch_size * seq_len)
        recon_loss = self.decoder.loss(_output_logits,  _tgt)
        # (batch_size, seq_len)
        recon_loss = recon_loss.view(batch_size, n_sample, -1)

        # (batch_size, 1)
        if recon_type=='avg':
            # avg over tokens
            recon_loss = recon_loss.mean(-1)
        elif recon_type=='sum':
            # sum over tokens
            recon_loss = recon_loss.sum(-1)
        elif recon_type == 'eos':
            # only eos token
            recon_loss = recon_loss[:,:,-1]

        # avg over batches and samples
        recon_acc  = self.accuracy(output_logits, tgt)
        return recon_loss, recon_acc

    def recon_loss_test(self, y, z, decoder_hidden, tag_attention_values, recon_type='avg'):
        #remove end symbol
        src = y[:, :-1]
        # remove start symbol
        tgt = y[:, 1:]        
        batch_size, seq_len = src.size()
        n_sample = z.size(1)

        decoder_input = src[:,0].unsqueeze(1)

        output_logits = []
        for di in range(seq_len):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, z, decoder_hidden, tag_attention_values)
            output_logits.append(decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()  # detach from history as input
        # (batchsize, seq_len, vocabsize)
        output_logits = torch.cat(output_logits,dim=1)

        _tgt = tgt.contiguous().view(-1)
        
        # (batch_size  * seq_len, vocab_size)
        _output_logits = output_logits.view(-1, output_logits.size(2))

        # (batch_size * seq_len)
        recon_loss = self.decoder.loss(_output_logits,  _tgt)
        # (batch_size, seq_len)
        recon_loss = recon_loss.view(batch_size, n_sample, -1)

        # (batch_size, 1)
        if recon_type=='avg':
            # avg over tokens
            recon_loss = recon_loss.mean(-1)
        elif recon_type=='sum':
            # sum over tokens
            recon_loss = recon_loss.sum(-1)
        elif recon_type == 'eos':
            # only eos token
            recon_loss = recon_loss[:,:,-1]

        # avg over batches and samples
        recon_acc, recon_preds  = self.accuracy(output_logits, tgt, mode='val')
        #breakpoint()
        return recon_loss, recon_acc, recon_preds

    def decode(self, x, tags, tmp):
        # a * [U(x)] + [Lu (xs|xt)] + [Ll (xt, yt| xs) - D(xt|yt)]
         # Ll (xt, yt | xs)
        mu, logvar, encoder_fhs = self.encoder(x)
        gumbel_tag_embeddings, xloss, tag_correct, tag_total, preds = self.classifier_loss(encoder_fhs, tmp)
        # (batchsize, 1, nz)
        z = mu.unsqueeze(0)
        
        tag_embeds = []
        for i in range(gumbel_tag_embeddings.shape[1]):
            #(batchsize, self.tag_embed_dim)
            tag_emb = gumbel_tag_embeddings[:,i,:]
            tag_embeds.append(tag_emb)
        # (batchsize, 11, self.tag_embed_dim)
        tag_embeddings = torch.stack(tag_embeds,dim=1)


        # (batchsize, 1, self.tag_embed_dim)
        tag_all_embed = torch.sum(tag_embeddings,dim=1).unsqueeze(1)
        #TODO: add bias
        tag_all_embed = torch.tanh(tag_all_embed)
        decoder_hidden = torch.tanh(self.tag_to_dec(tag_all_embed) + self.z_to_dec(z))
        decoder_hidden = torch.permute(decoder_hidden, (1,0,2))
     
        #### GREEDY DECODING
        decoder_input = torch.tensor(self.decoder.vocab.char2id["<s>"]).unsqueeze(0).unsqueeze(0).to('cuda')
        output_logits = []
        preds = []
        di = 0
        while True:
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, z, decoder_hidden, tag_embeddings)
            output_logits.append(decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()  # detach from history as input
            char = self.decoder.vocab.id2char[decoder_input.item()]
            preds.append(char)
            di +=1
            if di==50 or char == '</s>':
                break
        reinflected_form = ''.join(preds)
        return reinflected_form

    def sample(self):
        z = self.prior.sample((1,)).unsqueeze(0).to('cuda')
        tag_embeds = []
        for i in range(self.NUM_CODEBOOK):
            #(batchsize, self.tag_embed_dim)
            tag_emb = self.tag_embeddings[i](torch.tensor(2).to('cuda')).unsqueeze(0)
            tag_embeds.append(tag_emb)
        # (batchsize, NUM_CODEBOOK, self.tag_embed_dim)
        tag_embeddings = torch.stack(tag_embeds,dim=1)
        # (batchsize, 1, self.tag_embed_dim)
        tag_all_embed = torch.sum(tag_embeddings,dim=1).unsqueeze(1)
        #TODO: add bias
        tag_all_embed = torch.tanh(tag_all_embed)
        decoder_hidden = torch.tanh(self.tag_to_dec(tag_all_embed) + self.z_to_dec(z))
        decoder_hidden = torch.permute(decoder_hidden, (1,0,2))
        #### GREEDY DECODING
        decoder_input = torch.tensor(self.decoder.vocab.char2id["<s>"]).unsqueeze(0).unsqueeze(0).to('cuda')
        output_logits = []
        preds = []
        di = 0
        while True:
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, z, decoder_hidden, tag_embeddings)
            output_logits.append(decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()  # detach from history as input
            char = self.decoder.vocab.id2char[decoder_input.item()]
            preds.append(char)
            di +=1
            if di==50 or char == '</s>':
                break
        reinflected_form = ''.join(preds)
        return reinflected_form

        '''
        ### BEAM SEARCH DECODING
        K = 8
        decoded_batch = []
       
        # decoding goes sentence by sentence
        for idx in range(1):
            # Start with the start of the sentence token
            decoder_input = torch.tensor([[self.decoder.vocab.char2id["<s>"]]], dtype=torch.long, device='cuda')
            decoder_hidden = decoder_hidden[:,idx,:].unsqueeze(1)

            node = BeamSearchNode(decoder_hidden, None, decoder_input, 0., 1)
            live_hypotheses = [node]
            completed_hypotheses = []

            t = 0
            while len(completed_hypotheses) < K and t < 100:
                t += 1
                # (len(live), 1)
                decoder_input = torch.cat([node.wordid for node in live_hypotheses], dim=0)
                # (1, len(live), nh)
                decoder_hidden_h = torch.cat([node.h for node in live_hypotheses], dim=1)
                decoder_hidden = decoder_hidden_h


                #(len(live), 1, nz)
                expanded_z = z[idx].view(1, 1, -1).expand(len(live_hypotheses), 1, self.nz)
                expanded_tag_embeddings = tag_embeddings[idx].view(1, 11, -1).expand(len(live_hypotheses), 11, 200)
                expanded_tag_attention_masks = tag_attention_masks[idx].view(1, 1, -1).expand(len(live_hypotheses), 1, 11)
                output_logits, decoder_hidden, _ = self.decoder(decoder_input, expanded_z, decoder_hidden, expanded_tag_embeddings, expanded_tag_attention_masks)
                decoder_output = F.log_softmax(output_logits, dim=-1)

                prev_logp = torch.tensor([node.logp for node in live_hypotheses], dtype=torch.float, device='cuda')
                decoder_output = decoder_output + prev_logp.view(len(live_hypotheses), 1, 1)

                # (len(live) * vocab_size)
                decoder_output = decoder_output.view(-1)

                # (K)
                log_prob, indexes = torch.topk(decoder_output, K-len(completed_hypotheses))
                live_ids = indexes // len(self.decoder.vocab.char2id)
                word_ids = indexes % len(self.decoder.vocab.char2id)

                live_hypotheses_new = []
                for live_id, word_id, log_prob_ in zip(live_ids, word_ids, log_prob):
                    node = BeamSearchNode(decoder_hidden[:, live_id, :].unsqueeze(1),
                        live_hypotheses[live_id], word_id.view(1, 1), log_prob_, t)
                    if word_id.item() == self.decoder.vocab.char2id["<s>"]:
                        completed_hypotheses.append(node)
                    else:
                        live_hypotheses_new.append(node)
                live_hypotheses = live_hypotheses_new
                if len(completed_hypotheses) == K:
                    break
            for live in live_hypotheses:
                completed_hypotheses.append(live)
            utterances = []
            for n in sorted(completed_hypotheses, key=lambda node: node.logp, reverse=True):
                utterance = []
                utterance.append(self.decoder.vocab.id2char[n.wordid.item()])
                # back trace
                while n.prevNode != None:
                    n = n.prevNode
                    utterance.append(self.decoder.vocab.id2char[n.wordid.item()])
                utterance = utterance[::-1]
                utterances.append(utterance)
                # only save the top 1
                break
            decoded_batch.append(utterances[0])

            
        return ''.join(decoded_batch[0])'''

    def reparameterize(self, mu, logvar, nsamples=1):
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
        eps = torch.zeros_like(std_expd).normal_()
        return mu_expd + torch.mul(eps, std_expd)
    
    def accuracy(self, output_logits, tgt, mode='train'):
        # calculate correct number of predictions 
        batch_size, T = tgt.size()
        sft = nn.Softmax(dim=2)
        # (batchsize, T)
        pred_tokens = torch.argmax(sft(output_logits),2)
        acc = (pred_tokens == tgt).sum().item()
        if mode == 'val':
            return acc, pred_tokens
        else:
            return acc


    def reinflect(self, lemma, tgt, tmp):
        # Ll (xt, yt | xs)
        mu, logvar, _ = self.encoder(lemma)
        z = mu.unsqueeze(1)
        _, _, encoder_fhs = self.encoder(tgt)
        gumbel_tag_embeddings, xloss, tag_correct, tag_total, preds = self.classifier_loss(encoder_fhs, tmp)
        tag_embeds = []
        for i in range(gumbel_tag_embeddings.shape[1]):
            #(batchsize, self.tag_embed_dim)
            tag_emb = gumbel_tag_embeddings[:,i,:]
            tag_embeds.append(tag_emb)
        # (batchsize, 11, self.tag_embed_dim)
        tag_embeddings = torch.stack(tag_embeds,dim=1)

        # (batchsize, 1, self.tag_embed_dim)
        tag_all_embed = torch.sum(tag_embeddings,dim=1).unsqueeze(1)
        #TODO: add bias
        tag_all_embed = torch.tanh(tag_all_embed)
        decoder_hidden = torch.tanh(self.tag_to_dec(tag_all_embed) + self.z_to_dec(z))
        decoder_hidden = torch.permute(decoder_hidden, (1,0,2))
     
        #### GREEDY DECODING
        decoder_input = torch.tensor(self.decoder.vocab.char2id["<s>"]).unsqueeze(0).unsqueeze(0).to('cuda')
        output_logits = []
        preds = []
        di = 0
        while True:
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, z, decoder_hidden, tag_embeddings)
            output_logits.append(decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze(1).detach()  # detach from history as input
            char = self.decoder.vocab.id2char[decoder_input.item()]
            preds.append(char)
            di +=1
            if di==50 or char == '</s>':
                break
        reinflected_form = ''.join(preds)
        return reinflected_form



class BeamSearchNode(object):
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward
