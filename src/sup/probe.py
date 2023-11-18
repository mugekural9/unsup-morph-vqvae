import torch
import torch.nn as nn
import torch.nn.functional as F

class Probe(nn.Module):
    def __init__(self, Z_NH, tagsize):
        nn.Module.__init__(self)
        self.probe = nn.Linear(Z_NH, tagsize)
        self.loss = nn.CrossEntropyLoss(reduction='sum', ignore_index= 0)

    def forward(self, z: torch.Tensor, y: torch.Tensor):
        logits = self.probe(z) 
        xloss = self.loss(logits.squeeze(1), y.reshape(-1))
        # (batchsize,ty, vocabsize)
        sft= nn.Softmax(dim=2)
        probs = sft(logits)
        pred_tokens = torch.argmax(probs,dim=2)

        true = sum(pred_tokens==y.unsqueeze(1))
        total = len(y)
        return xloss, pred_tokens, true, total
