
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_EMB_DIM = 64
ENC_NH = 256
DEC_NH = 256
Z_NH = 32
VOCABSIZE = 51
ENC_DROPOUT_IN = 0.0
DEC_DROPOUT_IN = 0.3
MAX_LENGTH = 100

class AE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder_GRU_AE()
        if self.encoder.bidirectional:
            self.down_to_z_layer = nn.Linear(ENC_NH*2, Z_NH)
        else:
            self.down_to_z_layer = nn.Linear(ENC_NH, Z_NH)
        self.up_from_z_layer = nn.Linear(Z_NH, DEC_NH)

        self.decoder = Decoder_GRU_AE()
    
    def forward(self, x: torch.Tensor):
        out, out_h  = self.encoder(x)

        #(1,batchsize,zdim)
        z = self.down_to_z_layer(out_h)

        #(1,batchsize,dec_nh)
        hidden = self.up_from_z_layer(z)

        #(bathsize,1,zdim)
        z = torch.permute(z, (1,0,2)) 

        xloss, nonpadded_tokens, pred_tokens = self.decoder(x, hidden, z)
        totalloss =  xloss
        return totalloss, nonpadded_tokens, pred_tokens

    def decode(self, x: torch.Tensor):
        out, out_h  = self.encoder(x)
        # (1,1,32)
        z = self.down_to_z_layer(out_h)
        hidden = self.up_from_z_layer(z)
        #(bathsize,1,zdim)
        z = torch.permute(z, (1,0,2)) 
        pred_tokens = self.decoder.decode(x, hidden, z, MAX_LENGTH)
        return pred_tokens, z

class Encoder_GRU_AE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.input_dim = INPUT_EMB_DIM
        self.enc_nh = ENC_NH
        self.z_nh = Z_NH
        self.num_layers = 1
        self.vocabsize = VOCABSIZE 
        self.enc_dropout_in = ENC_DROPOUT_IN
        self.bidirectional = True

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
            # (1,batchsize, enc_nh*2)
            out_h = torch.cat([out_h[-2], out_h[-1]], 1).unsqueeze(0)
            out = (_,out_h)
        return out


class Decoder_GRU_AE(nn.Module):
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
        layers.append(nn.GRU(input_size=self.input_dim + self.z_incat_dim,
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
        out, hidden = self.layers[2](out, hidden)
        out = self.layers[1](out) #dropoutlayer

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
