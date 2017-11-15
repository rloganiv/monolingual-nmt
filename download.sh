#! /bin/bash
# Download datasets and preprocessing/evaluation scripts.

SRC_DIR=.
TOOL_DIR=tools
DATA_DIR=data

mkdir -p $TOOL_DIR
mkdir -p $DATA_DIR


# ---TOOLS---

# Moses decoder - tokenization, truecasing, BLEU evaluation script
if [ ! -d $TOOL_DIR/mosesdecoder ]; then
    git clone git@github.com:moses-smt/mosesdecoder.git $TOOL_DIR/mosesdecoder
fi

# # Byte pair encoding
# if [ ! -d $TOOL_DIR/subword-nmt ]; then
#     git clone git@github.com:rsennrich/subword-nmt.git $TOOL_DIR/subword-nmt
# fi
# 
# # Fast word2vec implementation
# if [ ! -d $TOOL_DIR/fastText ]; then
#     git clone git@github.com:facebookresearch/fastText $TOOL_DIR/fastText
# fi
# 
# # Map bilingual word vectors to a common space
# if [ ! -d $TOOL_DIR/vecmap ]; then
#     git clone git@github.com:artetxem/vecmap.git $TOOL_DIR/vecmap
# fi


# ---DATA---

# Pretrained cross lingual embeddings
EMBEDDING_DIR=$DATA_DIR/embeddings
mkdir -p $EMBEDDING_DIR

if [ ! -f $EMBEDDING_DIR/en.emb.txt ] || [ ! -f $EMBEDDING_DIR/fr.emb.txt ]; then
    echo "Downloading word embeddings."
    wget -O $EMBEDDING_DIR/embeddings.tar.gz https://www.dropbox.com/s/cygfsoomu6olhcc/embeddings.tar.gz?dl=1
    tar -xzf $EMBEDDING_DIR/embeddings.tar.gz -C $EMBEDDING_DIR
    rm $EMBEDDING_DIR/embeddings.tar.gz
fi
    

# # FR-EN Parrallel
# if [ ! -d $DATA_DIR/train-full ]; then
#     echo "Downloading the training datasets."
#     wget -O $DATA_DIR/training-parallel-europarl-v7.tgz http://statmt.org/wmt13/training-parallel-europarl-v7.tgz
#     wget -O $DATA_DIR/training-parallel-commoncrawl.tgz http://statmt.org/wmt13/training-parallel-commoncrawl.tgz
#     wget -O $DATA_DIR/training-parallel-un.tgz http://statmt.org/wmt13/training-parallel-un.tgz
#     wget -O $DATA_DIR/training-parallel-nc-v9.tgz http://statmt.org/wmt14/training-parallel-nc-v9.tgz
#     wget -O $DATA_DIR/training-parallel-giga-fren.tgz http://statmt.org/wmt10/training-giga-fren.tar
# fi
# 
# # Test Data
# if [ ! -d $DATA_DIR/test-full ]; then
#     echo "Downloading the test dataset."
#     wget -O $DATA_DIR/test-full.tgz http://statmt.org/wmt14/test-full.tgz
#     tar -xf $DATA_DIR/test-full.tgz
#     rm $DATA_DIR/test-full.tgz
#     rm $DATA_DIR/test-full/!(*fr*)
# fi
