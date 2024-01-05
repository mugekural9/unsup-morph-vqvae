
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
INPUT_EMB_DIM = 64
ENC_NH = 256
DEC_NH = 256
Z_NH = 32
#NUM_ENTRY = 2000

VOCABSIZE = 51
ENC_DROPOUT_IN = 0.0
DEC_DROPOUT_IN = 0.0
MAX_LENGTH = 100
COMMITMENT_LOSS_FACTOR = 0.1
QUANTIZATION_LOSS_FACTOR =0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

class VQVAE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.encoder = Encoder_GRU_VQVAE()
        if self.encoder.bidirectional:
            self.down_to_z_layer = nn.Linear(ENC_NH*2, Z_NH)
        self.quantizer = Quantizer()
        self.up_from_z_layer = nn.Linear(Z_NH, DEC_NH)

        self.decoder = Decoder_GRU_VQVAE()

    def forward(self, x: torch.Tensor):
        out, out_h  = self.encoder(x)
        
        #(1,batchsize,zdim)
        z = self.down_to_z_layer(out_h)
        #(quantized: 1,batchsize,dec_nh), (loss: batchsize)
        quantized_z, quantized_indices, qloss = self.quantizer(z)
        
        hidden = self.up_from_z_layer(quantized_z)
        quantized_z = torch.permute(quantized_z, (1,0,2)) 
        xloss, nonpadded_tokens, pred_tokens = self.decoder(x, hidden, quantized_z)
        totalloss =  xloss
        # sum Q loss over batch (per word)
        Q = torch.sum(qloss)
        return totalloss, nonpadded_tokens, pred_tokens, Q, quantized_indices

    def decode(self, x: torch.Tensor):
        out, out_h  = self.encoder(x)
        #(1,batchsize,zdim)
        z = self.down_to_z_layer(out_h)
        quantized_z, quantized_indices, qloss = self.quantizer(z)

        hidden = self.up_from_z_layer(quantized_z)
        quantized_z = torch.permute(quantized_z, (1,0,2)) 
        pred_tokens = self.decoder.decode(x, hidden, quantized_z, MAX_LENGTH)
        return pred_tokens

    

class Encoder_GRU_VQVAE(nn.Module):
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

class Quantizer(nn.Module):
    def __init__(self):

        nn.Module.__init__(self)
        self.embedding_dim = Z_NH
        #self.num_embeddings = NUM_ENTRY
        self.commitment_loss_factor   = COMMITMENT_LOSS_FACTOR 
        self.quantization_loss_factor = QUANTIZATION_LOSS_FACTOR 

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, h: torch.Tensor):
        ## squared L2 distance  (vector_x - vector_e)^2
        distances = (
            (h.reshape(-1, self.embedding_dim) ** 2).sum(dim=-1, keepdim=True)
            + (self.embeddings.weight ** 2).sum(dim=-1)
            - 2 * h.reshape(-1, self.embedding_dim) @ self.embeddings.weight.T)
        
        #hnorm = (h.reshape(-1, self.embedding_dim) ** 2).sum(dim=-1, keepdim=True)
        #entry_norms =  (self.embeddings.weight ** 2).sum(dim=-1)
        #if self.training:
        #    breakpoint()
        #    print(hnorm)
        #print(entry_norms)
        closest = distances.argmin(-1).unsqueeze(-1)
        #(32,1)
        quantized_indices = closest#.reshape(z.shape[0], z.shape[1], z.shape[2])
        #32,10
        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings)
            .type(torch.float)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings.weight
        #32,1,512
        quantized = quantized.reshape_as(h)

        # (batchsize, embdim) --> (batchsize)
        commitment_loss = F.mse_loss(quantized.detach().reshape(-1, self.embedding_dim), h.reshape(-1, self.embedding_dim),reduction="none").mean(dim=-1)
        embedding_loss = F.mse_loss(quantized.reshape(-1, self.embedding_dim), h.detach().reshape(-1, self.embedding_dim),reduction="none").mean(dim=-1)

        quantized = h + (quantized - h).detach()

        loss = (
            commitment_loss * self.commitment_loss_factor
            + embedding_loss * self.quantization_loss_factor
        )
        # (quantized: batchsize,1, dec_nh) , (quantized_indices: batchsize,1), (loss:scalar) 
        #print(quantized_indices)
        return quantized, quantized_indices, loss


class QuantizerEMA(nn.Module):
    def __init__(self):

        nn.Module.__init__(self)

        self.embedding_dim = Z_NH
        self.num_embeddings = NUM_ENTRY
        self.commitment_loss_factor = 0.1
        self.decay = 0.99

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

        self.register_buffer("cluster_size", torch.zeros(self.num_embeddings))

        self.ema_embed = nn.Parameter(
            torch.Tensor(self.num_embeddings, self.embedding_dim)
        )

        self.ema_embed.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, z: torch.Tensor):

        distances = (
            (z.reshape(-1, self.embedding_dim) ** 2).sum(dim=-1, keepdim=True)
            + (self.embeddings.weight ** 2).sum(dim=-1)
            - 2 * z.reshape(-1, self.embedding_dim) @ self.embeddings.weight.T
        )

        closest = distances.argmin(-1).unsqueeze(-1)
        quantized_indices = closest#.reshape(z.shape[0], z.shape[1], z.shape[2])

        one_hot_encoding = (
            F.one_hot(closest, num_classes=self.num_embeddings)
            .type(torch.float)
            .squeeze(1)
        )

        # quantization
        quantized = one_hot_encoding @ self.embeddings.weight
        quantized = quantized.reshape_as(z)

        if self.training:

            n_i = torch.sum(one_hot_encoding, dim=0)

            self.cluster_size = self.cluster_size * self.decay + n_i * (1 - self.decay)

            dw = one_hot_encoding.T @ z.reshape(-1, self.embedding_dim)

            self.ema_embed = nn.Parameter(
                self.ema_embed * self.decay + dw * (1 - self.decay)
            )

            n = torch.sum(self.cluster_size)

            self.cluster_size = (
                (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            )

            self.embeddings.weight = nn.Parameter(
                self.ema_embed / self.cluster_size.unsqueeze(-1)
            )

        commitment_loss = F.mse_loss(
            quantized.detach().reshape(-1, self.embedding_dim),
            z.reshape(-1, self.embedding_dim),
            reduction="mean",
        )

        quantized = z + (quantized - z).detach()

        loss = commitment_loss * self.commitment_loss_factor


        return quantized, quantized_indices, loss



class Decoder_GRU_VQVAE(nn.Module):
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
            # (out: batchsize,ty, dec_nh), (h)
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

    