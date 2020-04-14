# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

#
# Data preprocessing configuration
#

N_MONO=10000000  # number of monolingual sentences for each language
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs
METHOD=bpe
#METHOD=unigram
#
# Initialize tools and data paths
#

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/sp_data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
SRC_RAW=$MONO_PATH/all.en
TGT_RAW=$MONO_PATH/all.fr
CONCAT_RAW=$MONO_PATH/all.en-fr
MODEL=$MONO_PATH/$METHOD.model
FULL_VOCAB=$MONO_PATH/vocab.en-fr.$METHOD


#
# Download and install tools
#

# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download fastBPE
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"

# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"


#
# Download monolingual data
#

cd $MONO_PATH

cp $PWD/data/para/dev $PWD/sp_data/para
cp $PWD/data/mono/all.en $PWD/sp_data/mono/
cp $PWD/data/mono/all.fr $PWD/sp_data/mono/

#echo "Downloading English files..."
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
#
#echo "Downloading French files..."
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.fr.shuffled.gz
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.fr.shuffled.gz
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.fr.shuffled.gz
#wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.fr.shuffled.gz
#
## decompress monolingual data
#for FILENAME in news*gz; do
#  OUTPUT="${FILENAME::-3}"
#  if [ ! -f "$OUTPUT" ]; then
#    echo "Decompressing $FILENAME..."
#    gunzip -k $FILENAME
#  else
#    echo "$OUTPUT already decompressed."
#  fi
#done
#
## concatenate monolingual data files
#if ! [[ -f "$SRC_RAW" && -f "$TGT_RAW" ]]; then
#  echo "Concatenating monolingual data..."
#  cat $(ls news*en* | grep -v gz) | head -n $N_MONO > $SRC_RAW
#  cat $(ls news*fr* | grep -v gz) | head -n $N_MONO > $TGT_RAW
#fi
#echo "EN monolingual data concatenated in: $SRC_RAW"
#echo "FR monolingual data concatenated in: $TGT_RAW"

# check number of lines
if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your EN monolingual data."; exit; fi
if ! [[ "$(wc -l < $TGT_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your FR monolingual data."; exit; fi

# concatenate raw english and french
if [ ! -f "$CONCAT_RAW" ]; then
    echo "Merging en and fr data..."
    cat $SRC_RAW $TGT_RAW | shuf > $CONCAT_RAW
fi
echo "Merged en-fr data in: $CONCAT_RAW"


# train method model
if [ ! -f "$MODEL" ]; then
    echo "Training $METHOD model..."
    spm_train --input=$CONCAT_RAW --model_prefix=$METHOD --vocab_size=60000 --character_coverage=1.0 --model_type=$METHOD
fi
echo "$METHOD model trained in: $MODEL"


# apply model to get tokenized raw corpus
if ! [[ -f "$SRC_RAW.$METHOD" && -f "$TGT_RAW.$METHOD" ]]; then
    echo "Applying $METHOD model to src and tgt..."
    spm_encode --model=$MODEL --output_format=piece $SRC_RAW > $SRC_RAW.$METHOD
    spm_encode --model=$MODEL --output_format=piece $TGT_RAW > $TGT_RAW.$METHOD
fi
echo "$METHOD applied to src in: $SRC_RAW.$METHOD"
echo "$METHOD applied to tgt in: $TGT_RAW.$METHOD"


# merging model tokenized raw corpus
if [ ! -f $CONCAT_RAW.$METHOD ]; then
    echo "Merging tokenized src and tgt..."
    cat $SRC_RAW.$METHOD $TGT_RAW.$METHOD | shuf > $CONCAT_RAW.$METHOD
fi
echo "model tokenized corpus merged in: $CONCAT_RAW.$METHOD"


# extract vocabulary
if [ ! -f "$FULL_VOCAB" ]; then
    echo "Extracting vocabulary..."
    $FASTBPE getvocab $CONCAT_RAW.$METHOD > $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"


# binarize data
if ! [[ -f "$SRC_RAW.$METHOD.pth" && -f "$TGT_RAW.$METHOD.pth" ]]; then
  echo "Binarizing data..."
  $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_RAW.$METHOD
  $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_RAW.$METHOD
fi
echo "EN binarized data in: $SRC_RAW.$METHOD.pth"
echo "FR binarized data in: $TGT_RAW.$METHOD.pth"








#
# Download parallel data (for evaluation only)
#

#cd $PARA_PATH
#
#echo "Downloading parallel data..."
#wget -c http://data.statmt.org/wmt17/translation-task/dev.tgz
#
#echo "Extracting parallel data..."
#tar -xzf dev.tgz


SRC_VALID=$PARA_PATH/dev/newstest2013-ref.en
TGT_VALID=$PARA_PATH/dev/newstest2013-ref.fr
SRC_TEST=$PARA_PATH/dev/newstest2014-fren-src.en
TGT_TEST=$PARA_PATH/dev/newstest2014-fren-src.fr


# check valid and test files are here
if ! [[ -f "$SRC_VALID.sgm" ]]; then echo "$SRC_VALID.sgm is not found!"; exit; fi
if ! [[ -f "$TGT_VALID.sgm" ]]; then echo "$TGT_VALID.sgm is not found!"; exit; fi
if ! [[ -f "$SRC_TEST.sgm" ]]; then echo "$SRC_TEST.sgm is not found!"; exit; fi
if ! [[ -f "$TGT_TEST.sgm" ]]; then echo "$TGT_TEST.sgm is not found!"; exit; fi

echo "Reading from sgm valid and test data..."
$INPUT_FROM_SGM < $SRC_VALID.sgm | cat > $SRC_VALID
$INPUT_FROM_SGM < $TGT_VALID.sgm | cat > $TGT_VALID
$INPUT_FROM_SGM < $SRC_TEST.sgm | cat > $SRC_TEST
$INPUT_FROM_SGM < $TGT_TEST.sgm | cat > $TGT_TEST

echo "Applying model to valid and test files..."
spm_encode --model=$MODEL --output_format=piece $SRC_VALID > $SRC_VALID.$METHOD
spm_encode --model=$MODEL --output_format=piece $SRC_TEST > $SRC_TEST.$METHOD
spm_encode --model=$MODEL --output_format=piece $TGT_VALID > $TGT_VALID.$METHOD
spm_encode --model=$MODEL --output_format=piece $TGT_TEST > $TGT_TEST.$METHOD

echo "Binarizing data..."
rm -f $SRC_VALID.$METHOD.pth $TGT_VALID.$METHOD.pth $SRC_TEST.$METHOD.pth $TGT_TEST.$METHOD.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID.$METHOD
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID.$METHOD
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST.$METHOD
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST.$METHOD


#
# Train fastText on concatenated embeddings
#
if ! [[ -f "$CONCAT_RAW.$METHOD.vec" ]]; then
  echo "Training fastText on $CONCAT_RAW.$METHOD..."
  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_RAW.$METHOD -output $CONCAT_RAW.$METHOD
#  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -ws 5 -neg 10 -input $CONCAT_RAW.$METHOD -output $CONCAT_RAW.$METHOD
fi
echo "Cross-lingual embeddings in: $CONCAT_RAW.$METHOD.vec"


# train model
python main.py --exp_name ${METHOD}_test --transformer True --n_enc_layers 4 --n_dec_layers 4 --share_enc 3 --share_dec 3 --share_lang_emb True --share_output_emb True --langs 'en,fr' --n_mono -1 --mono_dataset "en:./sp_data/mono/all.en.${METHOD}.pth,,;fr:./sp_data/mono/all.fr.${METHOD}.pth,," --para_dataset "en-fr:,./sp_data/para/dev/newstest2013-ref.XX.${METHOD}.pth,./sp_data/para/dev/newstest2014-fren-src.XX.${METHOD}.pth" --mono_directions 'en,fr' --word_shuffle 3 --word_dropout 0.1 --word_blank 0.2 --pivo_directions 'fr-en-fr,en-fr-en' --pretrained_emb "./sp_data/mono/all.en-fr.${METHOD}.vec" --pretrained_out True --lambda_xe_mono '0:1,100000:0.1,300000:0' --lambda_xe_otfd 1 --otf_num_processes 30 --otf_sync_params_every 1000 --enc_optimizer adam,lr=0.0001 --epoch_size 500000 --stopping_criterion bleu_en_fr_valid,10