import torch

def prepare_batches(dataset, sort='tgt', batchsize=32, device='cuda'): 
    lemma_ids, tgt_ids, tags_ids = dataset.get_tokens()
    tgt_ids= sorted(tgt_ids, key=lambda x: len(x), reverse=True) #first one is the longest seq
    if sort == 'tgt':
        key_ids = tgt_ids
    num_seq = 0
    padded_batches = {}
    batchid = 0
    while num_seq < len(key_ids):
        padded_batches[batchid] = []
        batch_seqs = key_ids[batchsize*batchid :batchsize*(batchid+1)]
        num_seq += len(batch_seqs)
        max_length = max([len(w) for w in batch_seqs])
        for seq in batch_seqs:
            while len(seq) < max_length:
                seq +=  [0]
            padded_batches[batchid].append(seq)
        batchid+=1
    for batchid, batch in padded_batches.items():
        padded_batches[batchid] = torch.tensor(batch).to(device)
    #number of batches
    return padded_batches 

def prepare_batches_with_no_pad(dataset, batchsize, device='cuda'): 
    lemma_ids, tgt_ids, tags_ids = dataset.get_tokens()
    tgt_ids= sorted(tgt_ids, key=lambda x: len(x), reverse=True)
    tgt_with_lengths = dict()
    for tgt in tgt_ids:
        if len(tgt) not in tgt_with_lengths:
            tgt_with_lengths[len(tgt)] = []
        tgt_with_lengths[len(tgt)].append(tgt)

    batch_id = 0
    batches = dict()
    for key,val in tgt_with_lengths.items():
        if len(val) <= batchsize:
            batch = val
            batches[batch_id] = torch.tensor(val).to(device)
            batch_id +=1
        else:
            offset= 0
            while (batchsize*offset) < len(val):
                batch = val[batchsize*offset:(batchsize*offset)+batchsize]
                offset +=1
                batches[batch_id] = torch.tensor(batch).to(device)
                batch_id +=1

    #number of batches
    return batches 