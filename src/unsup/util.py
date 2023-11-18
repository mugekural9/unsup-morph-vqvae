import torch

def accuracy_on_batch(x, tgt):
    true_tokens =  torch.sum((x ==tgt) * (tgt!=0))
    all_tokens  =  torch.sum(tgt!=0)
    exact_match = 0
    for i in range(x.shape[0]):
        if (torch.sum(x[i] == tgt[i]) == x.shape[1]):
            exact_match +=1
    return true_tokens, all_tokens, exact_match

def decode_batch(x, charvocab):
    decoded =[]
    for i in range(x.shape[0]):
        tokenstr = ""
        for t in x[i]:
            tokenstr += charvocab.id2char[t.item()]
        decoded.append(tokenstr)
    return decoded
