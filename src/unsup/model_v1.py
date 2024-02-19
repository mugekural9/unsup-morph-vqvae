
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

ENC_DROPOUT_IN = 0.0
MAX_LENGTH = 100
COMMITMENT_LOSS_FACTOR = 0.1
QUANTIZATION_LOSS_FACTOR =0.2
device = "cuda" if torch.cuda.is_available() else "cpu"

class VQVAE(nn.Module):
    def __init__(self, vocabsize, dictmeta, ENC_NH, DEC_NH, Z_LEMMA_NH, Z_TAG_NH, DEC_DROPOUT_IN, INPUT_EMB_DIM, NUM_CODEBOOK, NUM_CODEBOOK_ENTRIES, BIDIRECTIONAL):
        nn.Module.__init__(self)
        self.encoder = Encoder_GRU_VQVAE(vocabsize, ENC_NH, INPUT_EMB_DIM, BIDIRECTIONAL)
        self.z_linears = nn.ModuleList()
        self.quantizers = nn.ModuleList()
        for i in range(NUM_CODEBOOK):
            if self.encoder.bidirectional:
                self.z_linears.append(nn.Linear(ENC_NH*2, Z_TAG_NH))
                self.z_lemma_mu = nn.Linear(ENC_NH*2, Z_LEMMA_NH)
                self.z_lemma_logvar = nn.Linear(ENC_NH*2, Z_LEMMA_NH)
            self.quantizers.append(Quantizer(Z_TAG_NH, NUM_CODEBOOK_ENTRIES))  
        self.up_from_z_layer = nn.Linear(Z_LEMMA_NH + Z_TAG_NH, DEC_NH)
        self.decoder = Decoder_GRU_VQVAE(vocabsize, DEC_NH, Z_LEMMA_NH, Z_TAG_NH, DEC_DROPOUT_IN, INPUT_EMB_DIM)
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
        if not self.training:
            lemma = mu
        # avg KL loss over batch (per word)
        KL = KL.mean()
        #if not self.training:
        #    lemma = mu
        quantized_z = []
        quantized_indices = []
        qloss = []
        for idx,quantizer in enumerate(self.quantizers):
            #(1,batchsize,zdim)
            z = self.z_linears[idx](out_h)
            #(quantized: 1,batchsize,dec_nh), (loss: batchsize)
            _quantized_z, _quantized_indices, _qloss = quantizer(z, None)
            quantized_z.append(_quantized_z)
            quantized_indices.append(_quantized_indices)
            qloss.append(_qloss)
        #(batchsize, num_codebooks)
        quantized_codes =  torch.stack(quantized_indices,dim=1).squeeze(2)
        tmp_quantized_z = quantized_z
        #sum over all codebooks
        #(1,batchsize,dec_nh)
        quantized_z = torch.sum(torch.stack(quantized_z),dim=0)
        #(batchsize)
        qloss = torch.sum(torch.stack(qloss),dim=0)
        hidden = self.up_from_z_layer(torch.cat((quantized_z,lemma),dim=2))
        quantized_z = torch.permute(quantized_z, (1,0,2)) 
        lemma = torch.permute(lemma, (1,0,2)) 
        xloss, nonpadded_tokens, pred_tokens = self.decoder(x, hidden, quantized_z, lemma)
        totalloss =  xloss
        # sum Q loss over batch (per word)
        Q = torch.sum(qloss)
        return totalloss, nonpadded_tokens, pred_tokens, Q, KL, quantized_indices, (lemma,tmp_quantized_z), quantized_codes

    def decode(self, x: torch.Tensor):
        out, out_h, fwd, bck  = self.encoder(x)
        #(1,batchsize,zdim)
        mu     = self.z_lemma_mu(out_h) 
        logvar = self.z_lemma_logvar(out_h) 
        #(z: 1,batchsize,zdim) (KL:batchsize)
        lemma, KL = self.encoder.reparameterize(mu,logvar)
        if not self.training:
            lemma = mu
        quantized_z = []
        quantized_indices = []
        for idx,quantizer in enumerate(self.quantizers):
            #(1,batchsize,zdim)
            z = self.z_linears[idx](out_h)
            #(quantized: 1,batchsize,dec_nh), (loss: batchsize)
            _quantized_z, _quantized_indices, _qloss  = quantizer(z,  None)
            quantized_z.append(_quantized_z)
            quantized_indices.append(_quantized_indices)
        #sum over all codebooks
        #(1,batchsize,dec_nh)
        quantized_z = torch.sum(torch.stack(quantized_z),dim=0)
        hidden = self.up_from_z_layer(torch.cat((quantized_z,lemma),dim=2))
        quantized_z = torch.permute(quantized_z, (1,0,2)) 
        lemma = torch.permute(lemma, (1,0,2)) 
        pred_tokens = self.decoder.decode(hidden, quantized_z, lemma, MAX_LENGTH)
        return pred_tokens, quantized_indices
    
    def sample(self):
        #(z:1,1,zdim) 
        lemma = self.prior.sample((1,)).unsqueeze(0).to('cuda')
        quantized_z = []
        quantized_indices = []
        for idx,quantizer in enumerate(self.quantizers):
            #(quantized: 1,batchsize,dec_nh),
            _quantized_z, _quantized_indices = quantizer.sample(torch.tensor(2))
            quantized_z.append(_quantized_z)
            quantized_indices.append(_quantized_indices)
        #sum over all codebooks
        #(1,batchsize,dec_nh)
        quantized_z = torch.sum(torch.stack(quantized_z),dim=0)
        hidden = self.up_from_z_layer(torch.cat((quantized_z,lemma),dim=2))
        quantized_z = torch.permute(quantized_z, (1,0,2)) 
        lemma = torch.permute(lemma, (1,0,2)) 
        pred_tokens = self.decoder.decode(hidden, quantized_z, lemma, MAX_LENGTH)
        return pred_tokens
    
    def reinflect(self, lemma, tgt, entries=None):
        _, out_lemma, _, _  = self.encoder(lemma)
        _, out_tgt, _, _  = self.encoder(tgt)
        #(1,batchsize,zdim)
        mu     = self.z_lemma_mu(out_lemma) 
        logvar = self.z_lemma_logvar(out_lemma) 
        #(z: 1,batchsize,zdim) (KL:batchsize)
        lemma, KL = self.encoder.reparameterize(mu,logvar)
        if not self.training:
            lemma = mu
        quantized_z = []
        quantized_indices = []
        for idx,quantizer in enumerate(self.quantizers):
            #(1,batchsize,zdim)
            z = self.z_linears[idx](out_tgt)
            #(quantized: 1,batchsize,dec_nh), (loss: batchsize)
            if entries is None:
                _quantized_z, _quantized_indices, _qloss = quantizer(z,  None)
            else:
                _quantized_z, _quantized_indices, _qloss = quantizer(z,  entries[idx].unsqueeze(0).to('cuda'))
            quantized_z.append(_quantized_z)
            quantized_indices.append(_quantized_indices)
        #sum over all codebooks
        #(1,batchsize,dec_nh)
        quantized_z = torch.sum(torch.stack(quantized_z),dim=0)
        hidden = self.up_from_z_layer(torch.cat((quantized_z,lemma),dim=2))
        quantized_z = torch.permute(quantized_z, (1,0,2)) 
        lemma = torch.permute(lemma, (1,0,2)) 
        pred_tokens = self.decoder.decode(hidden, quantized_z, lemma, MAX_LENGTH)
        return pred_tokens, quantized_indices
    
class Encoder_GRU_VQVAE(nn.Module):
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

class Quantizer(nn.Module):
    def __init__(self, dim_embeddings, num_embeddings):
        nn.Module.__init__(self)
        self.embedding_dim = dim_embeddings
        self.num_embeddings = num_embeddings
        self.commitment_loss_factor   = COMMITMENT_LOSS_FACTOR 
        self.quantization_loss_factor = QUANTIZATION_LOSS_FACTOR 
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(
            -1 / self.num_embeddings, 1 / self.num_embeddings
        )

    def forward(self, h: torch.Tensor, entry: torch.Tensor):
        ## squared L2 distance  (vector_x - vector_e)^2
        distances = (
            (h.reshape(-1, self.embedding_dim) ** 2).sum(dim=-1, keepdim=True)
            + (self.embeddings.weight ** 2).sum(dim=-1)
            - 2 * h.reshape(-1, self.embedding_dim) @ self.embeddings.weight.T)
        
        closest = distances.argmin(-1).unsqueeze(-1)
        if entry is not None:
            closest = entry.unsqueeze(1)

        #(32,1)
        quantized_indices = closest
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
    
    def sample(self, casted_entry:torch.Tensor):
        quantized_indices = casted_entry
        one_hot_encoding = (F.one_hot(casted_entry, num_classes=self.num_embeddings).type(torch.float)).to('cuda')
        # quantization
        quantized = one_hot_encoding @ self.embeddings.weight
        #1,1,64
        quantized = quantized.unsqueeze(0).unsqueeze(0)
        # (quantized: 1,1, dec_nh) , (quantized_indices: 1)
        return quantized, quantized_indices

class Decoder_GRU_VQVAE(nn.Module):
    def __init__(self, vocabsize, DEC_NH, Z_LEMMA_NH, Z_TAG_NH, DEC_DROPOUT_IN, INPUT_EMB_DIM):
        nn.Module.__init__(self)
        self.input_dim = INPUT_EMB_DIM
        self.z_incat_dim = Z_LEMMA_NH + Z_TAG_NH
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


    def forward(self, x: torch.Tensor, hidden: torch.Tensor, quantized_z: torch.Tensor, lemma: torch.Tensor):
        # (x:batchsize,ty+1), hidden(batchsize,1,512)
        # (src: batchsize,ty), (tgt: batchsize,ty)
        src = x[:,:-1]
        tgt = x[:,1:]
        #(bathsize,ty,quantized_zdim)
        quantized_z = quantized_z.repeat((1, tgt.size(1), 1))
        #(bathsize,ty,quantized_zdim)
        lemma = lemma.repeat((1, tgt.size(1), 1))
        #(batchsize,ty,inputdim)
        out = self.layers[0](src) #embeddinglayer
        out = self.layers[1](out) #dropoutlayer
        #(batchsize,ty,quantized_zdim+inputdim)
        out = torch.cat((quantized_z,out,lemma),dim=2)
        # (out: batchsize,ty, dec_nh), (h,c)
        out, hidden = self.layers[2](out, hidden)
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
    
    #one step at a time
    def forward_(self, x: torch.Tensor, hidden: torch.Tensor, quantized_z: torch.Tensor, lemma:torch.Tensor):
        # (x:batchsize,ty+1), hidden(batchsize,1,512)
        # (src: batchsize,ty), (tgt: batchsize,ty)
        # (quantized_z: batchsize,1,zdim), (lemma: batchsize,1,zdim)
        src = x[:,:-1]
        tgt = x[:,1:]
        pred_tokens = []
        xloss = 0
        nonpadded_tokens = 0
        # (batchsize,1)
        inputtoken = src[:,0].unsqueeze(1)
        for t in range(tgt.shape[1]):
            #(batchsize,1,input_emb_dim)
            out = self.layers[0](inputtoken)#embeddinglayer
            out = self.layers[1](out) #dropoutlayer
            #(batchsize,1,quantized_zdim+inputdim)
            out = torch.cat((quantized_z,out,lemma),dim=2)
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

    def decode(self, hidden: torch.Tensor, z: torch.Tensor, lemma: torch.Tensor, MAXLENGTH):
        # (1,1)
        inputtoken =torch.tensor(2).repeat(1,1).to('cuda') #BOS
        pred_tokens = []
        for t in range(MAXLENGTH):
            out = self.layers[0](inputtoken)#embeddinglayer
            #(batchsize,1,zdim+inputdim)
            out = torch.cat((z,out,lemma),dim=2)
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
