from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(["sp_data/mono/all.en-fr"], vocab_size=60000)

# with open("sp_data/mono/all.en-fr") as r, open("sp_data/mono/all.en-fr.wordpiece", "w") as w:
#     lines = r.readlines()
#     for line in lines:
#         encoded = tokenizer.encode(line[:-1])
#         w.write(" ".join(encoded.tokens))
#         w.write("\n")

with open("sp_data/para/dev/newstest2013-ref.en") as r, open("sp_data/para/dev/newstest2013-ref.en.bytebpe", "w") as w:
    lines = r.readlines()
    for line in lines:
        encoded = tokenizer.encode(line[:-1])
        w.write(" ".join(encoded.tokens))
        w.write("\n")

with open("sp_data/para/dev/newstest2013-ref.fr") as r, open("sp_data/para/dev/newstest2013-ref.fr.bytebpe", "w") as w:
    lines = r.readlines()
    for line in lines:
        encoded = tokenizer.encode(line[:-1])
        w.write(" ".join(encoded.tokens))
        w.write("\n")

with open("sp_data/para/dev/newstest2014-fren-src.en") as r, open("sp_data/para/dev/newstest2014-fren-src.en.bytebpe", "w") as w:
    lines = r.readlines()
    for line in lines:
        encoded = tokenizer.encode(line[:-1])
        w.write(" ".join(encoded.tokens))
        w.write("\n")

with open("sp_data/para/dev/newstest2014-fren-src.fr") as r, open("sp_data/para/dev/newstest2014-fren-src.fr.bytebpe", "w") as w:
    lines = r.readlines()
    for line in lines:
        encoded = tokenizer.encode(line[:-1])
        w.write(" ".join(encoded.tokens))
        w.write("\n")

with open("sp_data/mono/all.en") as r, open("sp_data/mono/all.en.bytebpe", "w") as w:
    lines = r.readlines()
    for line in lines:
        encoded = tokenizer.encode(line[:-1])
        w.write(" ".join(encoded.tokens))
        w.write("\n")

with open("sp_data/mono/all.fr") as r, open("sp_data/mono/all.fr.bytebpe", "w") as w:
    lines = r.readlines()
    for line in lines:
        encoded = tokenizer.encode(line[:-1])
        w.write(" ".join(encoded.tokens))
        w.write("\n")