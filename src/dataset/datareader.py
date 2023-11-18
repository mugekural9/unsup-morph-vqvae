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
    tids = []
    for t1,t2  in zip(tgt_ids,tags_ids):
        tids.append((t1,t2))

    tids = sorted(tids, key=lambda x: len(x[0]), reverse=True)
    t_with_lengths = dict()

    for t in tids:
        if len(t[0]) not in t_with_lengths:
            t_with_lengths[len(t[0])] = []
        t_with_lengths[len(t[0])].append(t)

    batch_id = 0
    batches = dict()
    for key,val in t_with_lengths.items():
        if len(val) <= batchsize:
            batches[batch_id] = [torch.tensor([v[0] for v in val]).to(device),  torch.tensor([v[1] for v in val]).to(device)]
            batch_id +=1
        else:
            offset= 0
            while (batchsize*offset) < len(val):
                batches[batch_id] = [torch.tensor([v[0] for v in val][batchsize*offset:(batchsize*offset)+batchsize]).to(device), 
                                     torch.tensor([v[1] for v in val][batchsize*offset:(batchsize*offset)+batchsize]).to(device)]
                offset +=1
                batch_id +=1
    #number of batches
    return batches

def prepare_batches_with_no_pad_wlemmas(dataset, batchsize, device='cuda'): 
    lemma_ids, tgt_ids, tags_ids = dataset.get_tokens()
    tids = []
    for t1,t2,t3  in zip(tgt_ids,tags_ids, lemma_ids):
        tids.append((t1,t2,t3))

    tids = sorted(tids, key=lambda x: len(x[0]), reverse=True)
    t_with_lengths = dict()

    for t in tids:
        if len(t[0]) not in t_with_lengths:
            t_with_lengths[len(t[0])] = []
        t_with_lengths[len(t[0])].append(t)

    batch_id = 0
    batches = dict()
    for key,val in t_with_lengths.items():
        if len(val) <= batchsize:
            batches[batch_id] = [torch.tensor([v[0] for v in val]).to(device),  
                                 torch.tensor([v[1] for v in val]).to(device), 
                                 torch.tensor([v[2] for v in val]).to(device)]
            batch_id +=1
        else:
            offset= 0
            while (batchsize*offset) < len(val):
                batches[batch_id] = [torch.tensor([v[0] for v in val][batchsize*offset:(batchsize*offset)+batchsize]).to(device), 
                                     torch.tensor([v[1] for v in val][batchsize*offset:(batchsize*offset)+batchsize]).to(device),
                                     torch.tensor([v[2] for v in val][batchsize*offset:(batchsize*offset)+batchsize]).to(device)]
                offset +=1
                batch_id +=1
    #number of batches
    return batches


"""def prepare_batches_with_no_pad_with_lemmas(dataset, batchsize, device='cuda'): 
    lemma_ids, tgt_ids, tags_ids = dataset.get_tokens()
    #tgt_ids= sorted(tgt_ids, key=lambda x: len(x), reverse=True)
    tgt_with_lengths = dict()
    lemma_with_lengths = dict()
    for lemma, tgt in zip(lemma_ids, tgt_ids):
        if len(tgt) not in tgt_with_lengths:
            tgt_with_lengths[len(tgt)] = []
            lemma_with_lengths[len(tgt)] = []
        tgt_with_lengths[len(tgt)].append(tgt)
        lemma_with_lengths[len(tgt)].append(lemma)

    batch_id = 0
    batches = dict()
    lemma_batches = dict()
    for key,val in tgt_with_lengths.items():
        lemma_val = lemma_with_lengths[key]
        if len(val) <= batchsize:
            batch = val
            batches[batch_id] = torch.tensor(val).to(device)
            lemma_batches[batch_id] = torch.tensor(lemma_val).to(device)

            batch_id +=1
        else:
            offset= 0
            while (batchsize*offset) < len(val):
                batch = val[batchsize*offset:(batchsize*offset)+batchsize]
                batch_lemma = lemma_val[batchsize*offset:(batchsize*offset)+batchsize]

                offset +=1
                batches[batch_id] = torch.tensor(batch).to(device)
                lemma_batches[batch_id] = torch.tensor(batch_lemma).to(device)
                batch_id +=1
    #number of batches
    return batches, lemma_batches"""