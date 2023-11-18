tag_id = "tense"
tag_values = ["PST", "FUT", "PRS"]
lines = []
with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/tur.gold") as reader:
    for line in reader:
        _line = line.strip()
        _, word, tags = _line.split('\t')
        for v in tag_values:
            if v in tags.split(";"):
                lines.append(line)

with open("/kuacc/users/mugekural/workfolder/dev/git/unsup-morph-vqvae/src/dataset/probe_filter_"+tag_id, "w") as writer:
    for line in lines:
        writer.write(line)

