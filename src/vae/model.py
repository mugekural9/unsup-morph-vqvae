
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
INPUT_EMB_DIM = 128
ENC_NH = 256
DEC_NH = 256
Z_NH = 32
VOCABSIZE = 51
ENC_DROPOUT_IN = 0.0
DEC_DROPOUT_IN = 0.2
MAX_LENGTH = 100
device = "cuda" if torch.cuda.is_available() else "cpu"

class VAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder_LSTM_VAE()
        self.decoder = Decoder_LSTM_VAE()

    def forward(self, x: torch.Tensor):
        z,h,c, KL = self.encoder(x)
        xloss, nonpadded_tokens, pred_tokens = self.decoder(x, (h, c), z)
        totalloss =  xloss
        # sum KL loss over batch (per word)
        KL = torch.sum(KL)
        return totalloss, nonpadded_tokens, pred_tokens, KL

    def pred(self, x: torch.Tensor):
        z,h,c = self.encoder(x)
        xloss, nonpadded_tokens, pred_tokens = self.decoder.forward_onestep_at_a_time(x, (h, c), z)
        totalloss =  xloss
        return totalloss, nonpadded_tokens, pred_tokens

    def decode(self, x: torch.Tensor):
        z,h,c, KL  = self.encoder(x)
        pred_tokens = self.decoder.decode(x, (h, c), z, MAX_LENGTH)
        return pred_tokens


    def sample(self, z):
     
        #(1,batchsize,dec_nh)
        c = self.encoder.up_from_z_layer(z)
        h = torch.tanh(c)
        hidden = (h,c)
        pred_tokens = self.decoder.sample(z, hidden, MAX_LENGTH)
        return pred_tokens


class Decoder_LSTM_VAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.input_dim = INPUT_EMB_DIM
        self.z_incat_dim = Z_NH
        
        self.dec_nh = DEC_NH
        self.num_layers = 1
        self.vocabsize = VOCABSIZE
        self.dec_dropout_in = DEC_DROPOUT_IN
        self.bidirectional = False
        layers = nn.ModuleList()
        layers.append(nn.Embedding(self.vocabsize, self.input_dim))
        layers.append(nn.Dropout(self.dec_dropout_in))
        layers.append(nn.LSTM(input_size=self.input_dim + self.z_incat_dim,
                            hidden_size=self.dec_nh,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional))
        layers.append(nn.Linear(self.dec_nh, self.vocabsize))
        self.layers = layers
        self.depth = len(layers)
        self.loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)


    def forward(self, x: torch.Tensor, hidden: torch.Tensor, z: torch.Tensor):
        # (x:batchsize,ty+1), hidden(batchsize,1,512)
        # (src: batchsize,ty), (tgt: batchsize,ty)
        src = x[:,:-1]
        tgt = x[:,1:]
        #(bathsize,ty,zdim)
        z = z.repeat((1, tgt.size(1), 1))
        #(batchsize,ty,inputdim)
        out = self.layers[0](src) #embeddinglayer
        out = self.layers[1](out) #dropoutlayer
        
        #(batchsize,ty,zdim+inputdim)
        out = torch.cat((z,out),dim=2)

        # (out: batchsize,ty, dec_nh), (h,c)
        out, (h,c) = self.layers[2](out, hidden)
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

    #_onestep_at_a_time
    def forward_onestep_at_a_time(self, x: torch.Tensor, hidden: torch.Tensor, z: torch.Tensor):
        # (x:batchsize,ty+1), hidden(batchsize,1,512)
        # (src: batchsize,ty), (tgt: batchsize,ty)
        src = x[:,:-1]
        tgt = x[:,1:]

        pred_tokens = []
        xloss = 0
        nonpadded_tokens = 0
        inputtoken = src[:,0].unsqueeze(1)
        for t in range(src.shape[1]):
            out = self.layers[0](inputtoken)#embeddinglayer
            out = self.layers[1](out) #dropoutlayer
            #(batchsize,1,zdim+inputdim)
            out = torch.cat((z,out),dim=2)
            # (out: batchsize,ty, dec_nh), (h,c)
            out, hidden = self.layers[2](out, hidden)
            # (batchsize,ty,vocabsize)
            logits = self.layers[3](out)
            xloss += self.loss(logits.reshape(-1,self.vocabsize), tgt[:,t].reshape(-1))
            nonpadded_tokens += torch.sum(tgt[:,t] != 0)
            #xloss/=nonpadded_tokens
            # (batchsize,ty,vocabsize)
            sft= nn.Softmax(dim=2)
            probs = sft(logits)
            # (batchsize,ty)
            inputtoken = torch.argmax(probs,dim=2)
            pred_tokens.append(inputtoken)

        pred_tokens = torch.stack(pred_tokens, dim=1).squeeze(-1)
        return xloss, nonpadded_tokens, pred_tokens


    def decode(self, x: torch.Tensor, hidden: torch.Tensor, z: torch.Tensor, MAXLENGTH):
        src = x[:,:-1]
        inputtoken = src[:,0].unsqueeze(1)
        pred_tokens = []
        for t in range(MAXLENGTH):
            out = self.layers[0](inputtoken)#embeddinglayer
            #(batchsize,1,zdim+inputdim)
            out = torch.cat((z,out),dim=2)
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

    def sample(self, z, hidden, MAXLENGTH):
        inputtoken = torch.full((1, 1), 2).to(device) #BOS:2
        pred_tokens = []
        for t in range(MAXLENGTH):
            out = self.layers[0](inputtoken)#embeddinglayer
            #(batchsize,1,zdim+inputdim)
            out = torch.cat((z,out),dim=2)
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



class Encoder_LSTM_VAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.input_dim = INPUT_EMB_DIM
        self.enc_nh = ENC_NH
        self.z_nh = Z_NH
        self.num_layers = 1
        self.vocabsize = VOCABSIZE 
        self.enc_dropout_in = ENC_DROPOUT_IN
        self.bidirectional = False

        layers = nn.ModuleList()
        layers.append(nn.Embedding(self.vocabsize, self.input_dim))
        layers.append(nn.Dropout(self.enc_dropout_in))
        layers.append(nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.enc_nh,
                            num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=self.bidirectional))

        self.down_to_mu_layer = nn.Linear(ENC_NH, Z_NH)
        self.down_to_std_layer = nn.Linear(ENC_NH, Z_NH)
        self.up_from_z_layer = nn.Linear(Z_NH, DEC_NH)
    
        self.layers = layers
        self.depth = len(layers)


    def forward(self, x: torch.Tensor):
        max_depth = self.depth
        out = x
        for i in range(max_depth):
            out = self.layers[i](out)

        # (out: batchsize,tx,enc_nh), (out_h: 1, batchsize, enc_nh), (out_c: 1, batchsize, enc_nh)
        out, (out_h, out_c) =  out
        # (1,batchsize,zdim)
        mu = self.down_to_mu_layer(out_h)
        # (1,batchsize,zdim)
        std = self.down_to_std_layer(out_h)
        # (1,batchsize,zdim)
        z, KL = self.reparameterize(mu, std)
        if not self.training:
            z = mu
        #(1,batchsize,dec_nh)
        c = self.up_from_z_layer(z)
        h = torch.tanh(c)
        #(bathsize,1,zdim)
        z = torch.permute(z, (1,0,2)) 
        return z,h,c,KL

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

