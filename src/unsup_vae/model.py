
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


ENC_DROPOUT_IN = 0.0
MAX_LENGTH = 100
COMMITMENT_LOSS_FACTOR = 0.1
QUANTIZATION_LOSS_FACTOR =0.2
device = "cuda" if torch.cuda.is_available() else "cpu"

class VAE(nn.Module):
    def __init__(self, vocabsize, ENC_NH, DEC_NH, Z_LEMMA_NH, DEC_DROPOUT_IN, INPUT_EMB_DIM, BIDIRECTIONAL):
        nn.Module.__init__(self)
        self.encoder = Encoder_GRU_VAE(vocabsize, ENC_NH, INPUT_EMB_DIM, BIDIRECTIONAL)
       
        if self.encoder.bidirectional:
            self.z_lemma_mu = nn.Linear(ENC_NH*2, Z_LEMMA_NH)
            self.z_lemma_logvar = nn.Linear(ENC_NH*2, Z_LEMMA_NH)
        else:
            self.z_lemma_mu = nn.Linear(ENC_NH, Z_LEMMA_NH)
            self.z_lemma_logvar = nn.Linear(ENC_NH, Z_LEMMA_NH)

        self.up_from_z_layer = nn.Linear(Z_LEMMA_NH , DEC_NH)
        self.decoder = Decoder_GRU_VAE(vocabsize, DEC_NH, Z_LEMMA_NH, DEC_DROPOUT_IN, INPUT_EMB_DIM)

        loc   = torch.zeros(Z_LEMMA_NH)
        scale = torch.ones(Z_LEMMA_NH)
        self.prior = torch.distributions.normal.Normal(loc, scale)

    def forward(self, x: torch.Tensor):
        out, out_h, fwd, bck  = self.encoder(x)
        #(1,batchsize,zdim)
        mu     = self.z_lemma_mu(out_h) 
        logvar = self.z_lemma_logvar(out_h) 
        #(z: 1,batchsize,zdim) (KL:batchsize)
        lemma, KL = self.encoder.reparameterize(mu,logvar)
        # avg KL loss over batch (per word)
        KL = KL.mean()

        #if not self.training:
        #    lemma = mu
        hidden = self.up_from_z_layer(lemma)
        lemma = torch.permute(lemma, (1,0,2)) 
        xloss, nonpadded_tokens, pred_tokens = self.decoder(x, hidden,  lemma)
        totalloss =  xloss
        return totalloss, nonpadded_tokens, pred_tokens, KL

    
    def decode(self, x: torch.Tensor):
        out, out_h, fwd, bck  = self.encoder(x)
        #(1,batchsize,zdim)
        mu     = self.z_lemma_mu(out_h) 
        logvar = self.z_lemma_logvar(out_h) 
        #(z: 1,batchsize,zdim) (KL:batchsize)
        lemma, KL = self.encoder.reparameterize(mu,logvar)
        
        #if not self.training:
        #    lemma = mu
        hidden = self.up_from_z_layer(lemma)
        lemma = torch.permute(lemma, (1,0,2)) 
        pred_tokens = self.decoder.decode(hidden, lemma, MAX_LENGTH)
        return pred_tokens
    
    
    def sample(self):
        #(z:1,1,zdim) 
        lemma = self.prior.sample((1,)).unsqueeze(0).to('cuda')
        hidden = self.up_from_z_layer(lemma)
        lemma = torch.permute(lemma, (1,0,2)) 
        pred_tokens = self.decoder.decode(hidden, lemma, MAX_LENGTH)
        return pred_tokens
    

class Encoder_GRU_VAE(nn.Module):
    def __init__(self, vocabsize, ENC_NH, INPUT_EMB_DIM, BIDIRECTIONAL):
        nn.Module.__init__(self)
        self.input_dim = INPUT_EMB_DIM
        self.enc_nh = ENC_NH
        self.num_layers = 1
        self.vocabsize = vocabsize 
        self.enc_dropout_in = ENC_DROPOUT_IN
        self.bidirectional = BIDIRECTIONAL

        layers = nn.ModuleList()
        layers.append(nn.Embedding(self.vocabsize, self.input_dim))
        layers.append(nn.Dropout(self.enc_dropout_in))
        layers.append(nn.GRU(input_size=self.input_dim,
                            hidden_size=self.enc_nh,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional))
        self.layers = layers
        self.depth = len(layers)


    def forward(self, x: torch.Tensor):
        max_depth = self.depth
        out = x
        for i in range(max_depth):
            out = self.layers[i](out)
        if self.bidirectional:
            # (out: batchsize, tx, enc_nh*2), (out_h: 2, batchsize,enc_nh)
            _, out_h = out
            # (1,batchsize, enc_nh)
            fwd = out_h[0].unsqueeze(0)
            bck = out_h[1].unsqueeze(0)
            out_h = torch.cat([out_h[0], out_h[1]], 1).unsqueeze(0)
        return _,out_h, fwd, bck
    

    def reparameterize(self, mu, logvar, nsamples=1):
        # KL: (batch_size), mu: (batch_size, nz), logvar: (batch_size, nz)
        mu = mu.squeeze(0) 
        logvar = logvar.squeeze(0) 
        KL = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1).sum(dim=1)
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()
        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)
        eps = torch.zeros_like(std_expd).normal_()
        z = mu_expd + torch.mul(eps, std_expd)
        z = torch.permute(z, (1,0,2))
        return z, KL


class Decoder_GRU_VAE(nn.Module):
    def __init__(self, vocabsize, DEC_NH, Z_LEMMA_NH, DEC_DROPOUT_IN, INPUT_EMB_DIM):
        nn.Module.__init__(self)
        self.input_dim = INPUT_EMB_DIM
        self.z_incat_dim = Z_LEMMA_NH 

        self.dec_nh = DEC_NH
        self.num_layers = 1
        self.vocabsize = vocabsize
        self.dec_dropout_in = DEC_DROPOUT_IN
        self.bidirectional = False
        layers = nn.ModuleList()
        layers.append(nn.Embedding(self.vocabsize, self.input_dim))
        layers.append(nn.Dropout(self.dec_dropout_in))
        layers.append(nn.GRU(input_size=self.input_dim + (self.z_incat_dim),
                            hidden_size=self.dec_nh,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional))
        layers.append(nn.Linear(self.dec_nh, self.vocabsize))
        self.layers = layers
        self.depth = len(layers)
        self.loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)


    def forward(self, x: torch.Tensor, hidden: torch.Tensor, lemma: torch.Tensor):
        # (x:batchsize,ty+1), hidden(batchsize,1,512)
        # (src: batchsize,ty), (tgt: batchsize,ty)
        src = x[:,:-1]
        tgt = x[:,1:]

        #(bathsize,ty,quantized_zdim)
        lemma = lemma.repeat((1, tgt.size(1), 1))

        #(batchsize,ty,inputdim)
        out = self.layers[0](src) #embeddinglayer
        out = self.layers[1](out) #dropoutlayer
        
        #(batchsize,ty,quantized_zdim+inputdim)
        out = torch.cat((out,lemma),dim=2)
        
        # (out: batchsize,ty, dec_nh), (h,c)
        out, hidden = self.layers[2](out, hidden)
        #out = self.layers[1](out) #dropoutlayer
        
        # (batchsize,ty,vocabsize)
        logits = self.layers[3](out)

        xloss = 0
        xloss = self.loss(logits.reshape(-1,self.vocabsize), tgt.reshape(-1))
        nonpadded_tokens = torch.sum(tgt != 0)
        #xloss/=nonpadded_tokens
        # (batchsize,ty,vocabsize)
        sft= nn.Softmax(dim=2)
        probs = sft(logits)
        # (batchsize,ty)
        pred_tokens = torch.argmax(probs,dim=2)
        return xloss, nonpadded_tokens, pred_tokens
    
    
    def decode(self, hidden: torch.Tensor, lemma: torch.Tensor, MAXLENGTH):
        # (1,1)
        inputtoken =torch.tensor(2).repeat(1,1).to('cuda') #BOS
        pred_tokens = []
        for t in range(MAXLENGTH):
            out = self.layers[0](inputtoken)#embeddinglayer
            #(batchsize,1,zdim+inputdim)
            out = torch.cat((out,lemma),dim=2)
            # (out: batchsize,ty, dec_nh), (h,c)
            out, hidden = self.layers[2](out, hidden)
            # (batchsize,ty,vocabsize)
            logits = self.layers[3](out)
            # (batchsize,ty,vocabsize)
            sft= nn.Softmax(dim=2)
            probs = sft(logits)
            # (batchsize,ty)
            inputtoken = torch.argmax(probs,dim=2)
            pred_tokens.append(inputtoken)
            if inputtoken.item() == 3: #EOS
                break
        pred_tokens = torch.stack(pred_tokens, dim=1).squeeze(-1)
        return pred_tokens
