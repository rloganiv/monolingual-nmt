"""Monolingual NMT Training Script"""

import argparse
import json
import logging
import math
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from numpy.random import sample
from torch.autograd import Variable

from evaluate import evaluate_autoencoder_model, evaluate_alignment_model, evaluate_mono_nmt
from model import Seq2SeqMono
from utils import load_embeddings, MonolingualDataset, MonolingualDataLoader

FLAGS = None


def load_config(path):
    """Loads the configuration.

    Args:
        path: Path to configuration file.
    """
    to_tuple = lambda d: namedtuple('X', d.keys())(*d.values())
    with open(path, 'r') as f:
        config = json.loads(f.read(), object_hook=to_tuple)
    return config


def hyperparam_string(config):
    """Hyerparam string."""
    exp_name = ''
    exp_name += 'model_%s__' % (config.data.task)
    exp_name += 'src_%s__' % (config.model.src_lang)
    exp_name += 'tgt_%s__' % (config.model.tgt_lang)
    exp_name += 'attention_%s__' % (config.model.seq2seq)
    exp_name += 'dim_%s__' % (config.model.dim)
    exp_name += 'emb_dim_%s__' % (config.model.dim_word_src)
    exp_name += 'optimizer_%s__' % (config.training.optimizer)
    exp_name += 'n_layers_src_%d__' % (config.model.n_layers_src)
    exp_name += 'n_layers_tgt_%d__' % (1)
    exp_name += 'bidir_%s' % (config.model.bidirectional)
    return exp_name


def initialize_logging(config):
    experiment_name = hyperparam_string(config)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='log/%s' % (experiment_name),
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)

    logging.getLogger('').addHandler(console)
    logging.info('Model Parameters : ')
    logging.info('Task : %s ' % (config.data.task))
    logging.info('Model : %s ' % (config.model.seq2seq))
    logging.info('Source Language : %s ' % (config.model.src_lang))
    logging.info('Target Language : %s ' % (config.model.tgt_lang))
    logging.info('Source Word Embedding Dim : %s' % (config.model.dim_word_src))
    logging.info('Target Word Embedding Dim : %s' % (config.model.dim_word_tgt))
    logging.info('Source RNN Hidden Dim : %s' % (config.model.dim))
    logging.info('Target RNN Hidden Dim : %s' % (config.model.dim))
    logging.info('Source RNN Depth : %d ' % (config.model.n_layers_src))
    logging.info('Target RNN Depth : %d ' % (1))
    logging.info('Source RNN Bidirectional : %s' % (config.model.bidirectional))
    logging.info('Batch Size : %d ' % (config.model.n_layers_tgt))
    logging.info('Optimizer : %s ' % (config.training.optimizer))
    logging.info('Learning Rate : %f ' % (config.training.lrate))


def trainable_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def train_ae(optimizer, criterion, model, ntokens, source, input_language,
             target, lengths, l1_decoder, use_maxlen, unsup):
    """Autoencoder training step."""

    optimizer.zero_grad()

    # Stupid hack needed for packing -_-
    lengths, index = torch.sort(lengths, dim=0, descending=True)
    source = source[index]
    target = target[index]

    output = model(source, target, lengths, input_language=input_language,
                   l1_decoder=l1_decoder, use_maxlen=use_maxlen, unsup=unsup)
    flattened_output = output.view(-1, ntokens)
    target = target.view(-1)
    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
    masked_output = \
        flattened_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion(masked_output, masked_target)
    loss.backward()
    optimizer.step()

    # Restore original order for output
    _, unindex = torch.sort(index)
    output = output[unindex]

    return loss, output


def train_bt(optimizer, criterion, model, ntokens, source,
             lengths, input_language, use_maxlen,
             unsup):
    """Backtranslation training step."""

    # Stupid hack needed for packing -_-
    lengths, index = torch.sort(lengths, dim=0, descending=True)
    source = source[index]

    if input_language == 'l1':
        l1_decoder = False # since we translate to l2
        backtranslation_input_language='l2'
    elif input_language == 'l2':
        l1_decoder = True
        backtranslation_input_language='l1'
    else:
        raise ValueError("input_language must be 'l1' or 'l2'")

    output = model(source, None, lengths, l1_decoder=l1_decoder,
                   input_language=input_language, use_maxlen=use_maxlen,
                   unsup=unsup)
    word_probs = model.decode(output, l1_decoder=l1_decoder)

    # Check axis
    _, predicted_sents = word_probs.max(dim=2)

    # TODO: Obtain lengths of decoded sentences...
    lengths = Variable(torch.LongTensor(predicted_sents.size()[0]).fill_(predicted_sents.size()[1]))
    lengths = lengths.cuda()

    # Feed predicted sents back through the model.

    loss, _ = train_ae(
        optimizer=optimizer,
        criterion=criterion,
        model=model,
        input_language=backtranslation_input_language,
        ntokens=ntokens,
        source=predicted_sents,
        target=source,
        lengths=lengths,
        l1_decoder=not l1_decoder, # since we're translating back
        use_maxlen=use_maxlen,
        unsup=unsup)

    return loss


# TODO: Refactor
# def save_model(epoch=None):
#     if epoch:
#         torch.save(model.state_dict(),
#                     open(os.path.join(save_dir,experiment_name + '__epoch_'+ str(epoch) + '_model.pt'), 'wb'))
#     else:
#         torch.save(model.state_dict(),
#                     open(os.path.join(save_dir,experiment_name + '_model.pt'), 'wb'))


def main(_):
    config = load_config(FLAGS.config)
    save_dir = config.data.save_dir
    load_dir = config.data.load_dir

    initialize_logging(config)

    # Load data
    logging.info('Reading embeddings')
    embeddings_src, vocab_src = load_embeddings(path=config.data.src_emb)
    embeddings_tgt, vocab_tgt = load_embeddings(path=config.data.tgt_emb)

    logging.info('Reading datasets')
    corpus_train_src = MonolingualDataset(folder=config.data.src_dir,
                                          train=True,
                                          vocab=vocab_src)
    corpus_train_tgt = MonolingualDataset(folder=config.data.trg_dir,
                                          train=True,
                                          vocab=vocab_tgt)
    l1_train_data = MonolingualDataLoader(dataset=corpus_train_src,
                                          batch_size=config.data.batch_size,
                                          shuffle=True)
    l2_train_data = MonolingualDataLoader(dataset=corpus_train_tgt,
                                          batch_size=config.data.batch_size,
                                          shuffle=True)
    logging.info('Done Reading')

    # Initialize model
    logging.info('Setting up model')

    gpuid = config.training.gpuid.strip().split(" ")
    gpuid = map(int, gpuid) if str(gpuid[0]) else None

    model = Seq2SeqMono(
        src_emb_dim=config.model.dim_word_src,
        tgt_emb_dim=config.model.dim_word_src,
        vocab_size_l1=len(vocab_src),
        vocab_size_l2=len(vocab_tgt),
        src_hidden_dim=config.model.dim,
        tgt_hidden_dim=config.model.dim,
        ctx_hidden_dim=config.model.dim,
        attention_mode='dot',
        batch_size=config.data.batch_size,
        bidirectional=config.model.bidirectional,
        nlayers=config.model.n_layers_src,
        nlayers_tgt=config.model.n_layers_tgt,
        maxlen = config.data.max_len-1,
        dropout=0.3,
        gpu=(gpuid is not None))

    model.init_weights(embeddings_src, embeddings_tgt)

    if gpuid and len(gpuid)>1:
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpuid)
    elif gpuid and len(gpuid) == 1:
        model = model.cuda()


    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params(model),
                           lr=config.training.lrate,
                           betas=(0.5, 0.999))
    # TODO: Refactor
    # if load_dir:
    #     model.load_state_dict(torch.load(open(load_dir+"/model.pt")))

    logging.info('Starting training')

    for i in xrange(1000): # Epochs

        losses = []
        l1_train_iter = iter(l1_train_data)
        l2_train_iter = iter(l2_train_data)

        for j in xrange(len(l1_train_iter)): # Training steps

            l1_sample = next(l1_train_iter)
            l2_sample = next(l2_train_iter)

            if gpuid:
                l1_sample = {k: v.cuda() for k, v in l1_sample.items()}
                l2_sample = {k: v.cuda() for k, v in l2_sample.items()}

            if j % 2: # Denoising step

                loss_l1, output_l1 = train_ae(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    input_language='l1',
                    ntokens=len(vocab_src),
                    source=l1_sample['src'],
                    target=l1_sample['tgt'],
                    lengths=l1_sample['src_len'],
                    l1_decoder=True,
                    use_maxlen=False,
                    unsup=True)

                loss_l2, output_l2 = train_ae(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    input_language='l2',
                    ntokens=len(vocab_src),
                    source=l2_sample['src'],
                    target=l2_sample['trg'],
                    lengths=l2_sample['src_len'],
                    l2_decoder=True,
                    use_maxlen=False,
                    unsup=True)

            else: # Backtranslation step

                loss_l1_l2 = train_bt(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    ntokens=len(vocab_src),
                    source=l1_sample['tgt'], # tgt because not distorted
                    lengths=l1_sample['tgt_len'],
                    input_language='l1',
                    use_maxlen=False,
                    unsup=True)

                loss_l2_l1 = train_bt(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    ntokens=len(vocab_src),
                    source=l2_sample['tgt'], # tgt because not distorted
                    lengths=l2_sample['tgt_len'],
                    input_language='l2',
                    use_maxlen=False,
                    unsup=True)

            losses.append((loss_l1.data[0], loss_l2.data[0], loss_l1_l2.data[0], loss_l2_l1.data[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config", required=True)
    FLAGS, _ = parser.parse_known_args()

    main(_)

