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
from model import Model
from utils import load_config, load_embeddings, greedy_translate
from utils import MonolingualDataset, MonolingualDataLoader

FLAGS = None
USE_CUDA = torch.cuda.is_available()


def hyperparam_string(config):
    """String detailing experiment hyperparameters."""
    exp_name = ''
    exp_name += 'task_%s__' % config.data.task
    exp_name += 'l1_%s__' % config.data.l1_language
    exp_name += 'l2_%s__' % config.data.l2_language
    exp_name += 'optimizer_%s__' % config.training.optimizer
    exp_name += 'backtranslate_%s__' % config.training.backtranslate
    exp_name += 'rnn_%s__' % config.model.rnn
    exp_name += 'embedding_dim_%s__' % config.model.embedding_dim
    exp_name += 'hidden_dim_%s__' % config.model.hidden_dim
    exp_name += 'n_layers_enc_%d__' % config.model.n_layers_encoder
    exp_name += 'n_layers_dec_%d' % config.model.n_layers_decoder
    return exp_name


def initialize_logging(config):
    """Initializes logging and prints experiment setup to the log."""
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
    logging.info('Task : %s ' % config.data.task)
    logging.info('Language l1 : %s ' % config.data.l1_language)
    logging.info('Language l2 : %s ' % config.data.l2_language)
    logging.info('Embedding Dim : %s' % config.model.embedding_dim)
    logging.info('Hidden Dim : %s' % config.model.hidden_dim)
    logging.info('Dropout Rate : %0.2f' % config.model.dropout)
    logging.info('Num. Layers in Encoder : %d ' % config.model.n_layers_encoder)
    logging.info('Num. Layers in Decoder : %d ' % config.model.n_layers_decoder)
    logging.info('Batch Size : %d ' % config.data.batch_size)
    logging.info('Optimizer : %s ' % config.training.optimizer)
    logging.info('Learning Rate : %f ' % config.training.lrate)
    logging.info('Backtranslate : %s ' % config.training.backtranslate)


def trainable_params(model):
    """Returns only the trainable parameters of a model."""
    return filter(lambda p: p.requires_grad, model.parameters())


def transform_inputs(src, lengths, tgt, transpose=True, add_dim=True):
    """Performs length sorting/dimension additions needed to feed input into
    encoder.

    Args:
        src: (LongTensor) len x batch_size x (opt. nfeats). Source sentences.
        lengths: (LongTensor) batch_size. Lengths of source sentences.
        tgt: (LongTensor) len x batch_size x (opt. nfeats). Target sentences.
        transpose: Whether to transpose first two dimensions of the data (e.g.
            batch size comes first.)
    """

    if add_dim:
        # Add 'nfeats' dim for OpenNMT compatibility.
        src = torch.unsqueeze(src, 2)
        tgt = torch.unsqueeze(tgt, 2)

    # Stupid hack since input dimensions are in the wrong order.
    if transpose:
        src = torch.transpose(src, 0, 1).contiguous()
        tgt = torch.transpose(tgt, 0, 1).contiguous()

    # Stupid hack needed for packing -_-
    lengths, index = torch.sort(lengths, dim=0, descending=True)
    src = src[:,index,:]
    tgt = tgt[:,index,:]

    return src, lengths, tgt, index


def train_step(optimizer, criterion, model, src, src_lang, lengths, tgt, tgt_lang):
    """Autoencoder training step."""

    optimizer.zero_grad()

    # Get vocab size.
    if tgt_lang == 'l1':
        output_vocab_size = model.l1_vocab_size
    elif tgt_lang == 'l2':
        output_vocab_size = model.l2_vocab_size
    else:
        raise ValueError('tgt_lang')

    # Model output.
    logits, _, _ = model(src=src, src_lang=src_lang, lengths=lengths, tgt=tgt,
                         tgt_lang=tgt_lang)

    # Fancy masking operations.
    flattened = logits.view(-1, output_vocab_size)

    tgt = tgt[1:].view(-1)
    mask = tgt.gt(0)
    masked_tgt = tgt.masked_select(mask)

    logit_mask = mask.unsqueeze(1).expand(mask.size(0), output_vocab_size)
    masked_logits = flattened.masked_select(logit_mask).view(-1, output_vocab_size)

    # Loss + backprop.
    loss = criterion(masked_logits, masked_tgt)
    loss.backward()
    optimizer.step()

    return loss, logits


def main(_):
    # Load configuration
    config = load_config(FLAGS.config)
    initialize_logging(config)

    # Load data
    logging.info('Reading embeddings')
    l1_embeddings, l1_vocab = load_embeddings(path=config.data.l1_embeddings)
    l2_embeddings, l2_vocab = load_embeddings(path=config.data.l2_embeddings)

    logging.info('Reading datasets')
    l1_dataset = MonolingualDataset(folder=config.data.l1_train_data,
                                    train=True,
                                    vocab=l1_vocab)
    l2_dataset = MonolingualDataset(folder=config.data.l2_train_data,
                                    train=True,
                                    vocab=l2_vocab)
    l1_dataloader = MonolingualDataLoader(dataset=l1_dataset,
                                          batch_size=config.data.batch_size,
                                          shuffle=True)
    l2_dataloader = MonolingualDataLoader(dataset=l2_dataset,
                                          batch_size=config.data.batch_size,
                                          shuffle=True)

    # Initialize model
    logging.info('Setting up model')

    gpuid = config.training.gpuid.strip().split(" ")
    gpuid = map(int, gpuid) if str(gpuid[0]) else None

    model_path = os.path.join(config.data.ckpt, 'model.pt')
    if os.path.exists(model_path):
        logging.info('Loading existing checkpoint at: %s' % model_path)
        model = torch.load(model_path)
    else:
        logging.info('Building from scratch')
        model = Model(
            l1_vocab_size=config.data.l1_vocab_size,
            l2_vocab_size=config.data.l2_vocab_size,
            rnn_type=config.model.rnn,
            embedding_size=config.model.embedding_dim,
            hidden_size=config.model.hidden_dim,
            n_layers_encoder=config.model.n_layers_encoder,
            n_layers_decoder=config.model.n_layers_decoder,
            dropout=config.model.dropout)
        model.l1_embeddings.load(l1_embeddings)
        model.l2_embeddings.load(l2_embeddings)

    if gpuid and len(gpuid)>1:
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpuid)
    elif gpuid and len(gpuid) == 1:
        logging.info('Moving model to GPU')
        model = model.cuda()

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(trainable_params(model),
                           lr=config.training.lrate,
                           betas=(0.5, 0.999))

    logging.info('Starting training')

    iters = 0

    for i in xrange(1000): # Epochs

        denoising_losses = []
        backtranslation_losses = []
        l1_iter = iter(l1_dataloader)
        l2_iter = iter(l2_dataloader)

        for _ in xrange(len(l1_iter)): # Training steps
            iters += 1

            l1_sample = next(l1_iter)
            l2_sample = next(l2_iter)

            if gpuid:
                l1_sample = {k: v.cuda() for k, v in l1_sample.items()}
                l2_sample = {k: v.cuda() for k, v in l2_sample.items()}

            if (not iters % 2) and (config.training.backtranslate): # Backtranslation step

                # Translate l1 to l2 then train on backtranslation from output
                # back to l1
                l1_src, l1_lengths, _, l1_index = transform_inputs(
                    src=l1_sample['tgt'], # Use target since undistorted.
                    lengths=l1_sample['tgt_len'].data,
                    tgt=l1_sample['tgt'])
                l1_to_l2, l1_to_l2_len = greedy_translate(
                    model=model,
                    src=l1_src,
                    src_lang='l1',
                    lengths=l1_lengths,
                    tgt_lang='l2',
                    max_length=config.data.max_length)
                l1_to_l2, l1_to_l2_len, l1_src, l1_to_l2_index = transform_inputs(
                    src=l1_to_l2,
                    lengths=l1_to_l2_len,
                    tgt=l1_src,
                    transpose=False,
                    add_dim=False)
                l2_to_l1_loss, l2_to_l1_logits = train_step(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    src=l1_to_l2,
                    src_lang='l2',
                    lengths=l1_to_l2_len,
                    tgt=l1_src,
                    tgt_lang='l1')

                # Translate l2 to l1 then train on backtranslation from output
                # back to l2
                l2_src, l2_lengths, _, l2_index = transform_inputs(
                    src=l2_sample['tgt'], # Use target since undistorted.
                    lengths=l2_sample['tgt_len'].data,
                    tgt=l2_sample['tgt'])
                l2_to_l1, l2_to_l1_len = greedy_translate(
                    model=model,
                    src=l2_src,
                    src_lang='l2',
                    lengths=l2_lengths,
                    tgt_lang='l1',
                    max_length=config.data.max_length)
                l2_to_l1, l2_to_l1_len, l2_src, l2_to_l1_index = transform_inputs(
                    src=l2_to_l1,
                    lengths=l2_to_l1_len,
                    tgt=l2_src,
                    transpose=False,
                    add_dim=False)
                l1_to_l2_loss, l1_to_l2_logits = train_step(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    src=l2_to_l1,
                    src_lang='l1',
                    lengths=l2_to_l1_len,
                    tgt=l2_src,
                    tgt_lang='l2')

                backtranslation_losses.append(
                    (l1_to_l2_loss.data[0],
                     l2_to_l1_loss.data[0]))

            else: # Denoising step
                l1_src, l1_lengths, l1_tgt, l1_index = transform_inputs(
                    l1_sample['src'],
                    l1_sample['src_len'].data,
                    l1_sample['tgt'])
                l1_loss, l1_output = train_step(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    src=l1_src,
                    src_lang='l1',
                    lengths=l1_lengths,
                    tgt=l1_tgt,
                    tgt_lang='l1')

                l2_src, l2_lengths, l2_tgt, l2_index = transform_inputs(
                    l2_sample['src'],
                    l2_sample['src_len'].data,
                    l2_sample['tgt'])
                l2_loss, l2_output = train_step(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    src=l2_src,
                    src_lang='l2',
                    lengths=l2_lengths,
                    tgt=l2_tgt,
                    tgt_lang='l2')

                denoising_losses.append((l1_loss.data[0], l2_loss.data[0]))

            if not iters % 100:
                l1_denoising_loss, l2_denoising_loss = zip(*denoising_losses)
                logging.info('Iteration: %i, L1 Denoising Loss: %0.4f' % (iters, np.mean(l1_denoising_loss)))
                logging.info('Iteration: %i, L2 Denoising Loss: %0.4f' % (iters, np.mean(l2_denoising_loss)))
                denoising_losses = []

                if config.training.backtranslate:
                    print l2_to_l1.size()
                    print l2_to_l1
                    l1_to_l2_bt_loss, l2_to_l1_bt_loss = zip(*backtranslation_losses)
                    logging.info('Iteration: %i, L1 to L2 Backtranslation Loss: %0.4f' % (iters, np.mean(l1_to_l2_bt_loss)))
                    logging.info('Iteration: %i, L2 to L1 Backtranslation Loss: %0.4f' % (iters, np.mean(l2_to_l1_bt_loss)))
                    backtranslation_losses = []

            if not iters % 1000:
                logging.info('Saving model')
                torch.save(model, os.path.join(config.data.ckpt, 'model.pt'))

            if not iters % config.training.stop_iter:
                logging.info('Training complete')
                torch.save(model, os.path.join(config.data.ckpt, 'model.pt'))
                sys.exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config", required=True)
    FLAGS, _ = parser.parse_known_args()

    main(_)

