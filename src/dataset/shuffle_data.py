import random
lines = []
with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata_NOUNS") as reader:
    for line in reader:
        lines.append(line)
random.Random(4).shuffle(lines)

with open("/kuacc/users/mugekural/workfolder/dev/tur/tur_filtered_duplicates_merged_stdata_NOUNS_shuffled","w") as writer:
    for line  in lines:
        writer.write(line)
