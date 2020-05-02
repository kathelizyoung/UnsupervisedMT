import numpy as np

bpe_log = "dumped/bpe_test/49xr4vceiz/train.log"
bytebpe_log = "dumped/bytebpe_test/4aa8lgchlf/train.log"
unigram_log = "dumped/unigram_test/kowdt1lwgx/train.log"
wordpiece_log = "dumped/wordpiece_test/lbgamxguj1/train.log"

wordpiece_test = []
with open(wordpiece_log) as f:
    for line in f.readlines():
        if "bleu_en_fr_test ->" in line:
            wordpiece_test.append(float(line.split("->")[1].strip()))

bpe_test = []
with open(bpe_log) as f:
    for line in f.readlines():
        if "bleu_en_fr_test ->" in line:
            bpe_test.append(float(line.split("->")[1].strip()))

bytebpe_test = []
with open(bytebpe_log) as f:
    for line in f.readlines():
        if "bleu_en_fr_test ->" in line:
            bytebpe_test.append(float(line.split("->")[1].strip()))

unigram_test = []
with open(unigram_log) as f:
    for line in f.readlines():
        if "bleu_en_fr_test ->" in line:
            unigram_test.append(float(line.split("->")[1].strip()))

all_results = {"WordPiece": wordpiece_test, "BPE": bpe_test, "ByteBPE": bytebpe_test, "Unigram": unigram_test}

# Best blue score of bpe:
# 25.4 at epoch 115
best_idx = np.argmax(bpe_test)
print("Best blue score of bpe:\n{0:0.2f} at epoch {1:d}".format(bpe_test[best_idx], best_idx))


# Best blue score of wordpiece:
# 29.66 at epoch 95
best_idx = np.argmax(wordpiece_test)
print("Best blue score of wordpiece:\n{0:0.2f} at epoch {1:d}".format(wordpiece_test[best_idx], best_idx))

best_idx = np.argmax(unigram_test)
print("Best blue score of unigram:\n{0:0.2f} at epoch {1:d}".format(unigram_test[best_idx], best_idx))

best_idx = np.argmax(bytebpe_test)
print("Best blue score of bytebpe:\n{0:0.2f} at epoch {1:d}".format(bytebpe_test[best_idx], best_idx))


interval = 5
length = max([len(all_results[method]) for method in all_results]) // interval
axis = [str(i*interval) for i in range(length)]
lample = ["25.10"] * len(axis)
with open("to_sheet.txt", "w") as f:

    f.write("-" + "\t" + "\t".join(axis) + "\n")

    for method in all_results:
        result = all_results[method]
        # print(result)
        length = len(result)
        count = length // interval
        tmp = []
        f.write(method+"\t")
        for i in range(count):
            tmp.append("{0:0.2f}".format(result[i*interval]))
        f.write("\t".join(tmp))
        f.write("\n")

    f.write("Lample" + "\t" + "\t".join(lample) + "\n")

# print(bpe_test[47])