#! /bin/bash
# Download datasets and preprocessing/evaluation scripts.

SRC_DIR=.
TOOL_DIR=tools

# Download tools
mkdir -p $SRC_DIR/$TOOL_DIR

# Moses decoder - tokenization, truecasing, BLEU evaluation script
git clone git@github.com:moses-smt/mosesdecoder.git $SRC_DIR/$TOOL_DIR/mosesdecoder

# Byte pair encoding
git clone git@github.com:rsennrich/subword-nmt.git $SRC_DIR/$TOOL_DIR/subword-nmt

# Fast word2vec implementation
git clone git@github.com:facebookresearch/fastText $SRC_DIR/$TOOL_DIR/fastText

# Map bilingual word vectors to a common space
git clone git@github.com:artetxem/vecmap.git $SRC_DIR/$TOOL_DIR/vecmap
