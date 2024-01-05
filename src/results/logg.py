with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/results/vqvae/unsup/nouns/6x16_zLEM100_zTAG128_decnh1024_kl1.0_epc10_strt5_decdo0.2_inpemb128_bsize128/training_vqvae.log", "r") as reader:
    trn_usage = []
    dev_usage = []
    i=0
    for line in reader:
        if "|| unique taglist code:" in line:
            if i%2==0:
                trn_usage.append(int(line.split('unique taglist code:')[1].strip()))
            else:
                dev_usage.append(int(line.split('unique taglist code:')[1].strip()))
            i+=1

print(trn_usage)
