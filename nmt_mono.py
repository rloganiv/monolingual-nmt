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
from model_remake import Model
from utils import load_embeddings, MonolingualDataset, MonolingualDataLoader

FLAGS = None
USE_CUDA = torch.cuda.is_available()


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
    exp_name += 'task_%s__' % (config.data.task)
    exp_name += 'l1_%s__' % (config.data.l1_language)
    exp_name += 'l2_%s__' % (config.data.l2_language)
    exp_name += 'optimizer_%s__' % (config.training.optimizer)
    exp_name += 'rnn_%s__' % (config.model.rnn)
    exp_name += 'embedding_dim_%s__' % (config.model.embedding_dim)
    exp_name += 'hidden_dim_%s__' % (config.model.hidden_dim)
    exp_name += 'n_layers_enc_%d__' % (config.model.n_layers_encoder)
    exp_name += 'n_layers_dec_%d' % (config.model.n_layers_decoder)
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


def trainable_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


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

    # Stupid hack needed for packing -_-
    lengths, index = torch.sort(lengths, dim=0, descending=True)
    src = src[index]
    tgt = tgt[index]

    # Stupid hack since input dimensions are in the wrong order.
    src = torch.transpose(src, 0, 1).contiguous()
    tgt = torch.transpose(tgt, 0, 1).contiguous()

    # Model output.
    logits, _, _ = model(src=src, src_lang=src_lang, lengths=lengths, tgt=tgt,
                         tgt_lang=tgt_lang)

    # Fancy masking operations.
    flattened = logits.view(-1, output_vocab_size)

    tgt = tgt[:-1].view(-1) # Since the last output is omitted by the model.
    mask = tgt.gt(0)
    masked_tgt = tgt.masked_select(mask)

    logit_mask = mask.unsqueeze(1).expand(mask.size(0), output_vocab_size)
    masked_logits = flattened.masked_select(logit_mask).view(-1, output_vocab_size)

    # Loss + backprop.
    loss = criterion(masked_logits, masked_tgt)
    loss.backward()
    optimizer.step()

    # Restore original order for output.
    _, unindex = torch.sort(index)
    logits = torch.transpose(logits, 0, 1)
    logits = logits[unindex]

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
    # TODO: Refactor
    # if load_dir:
    #     model.load_state_dict(torch.load(open(load_dir+"/model.pt")))

    logging.info('Starting training')

    iters = 0

    for i in xrange(1000): # Epochs

        denoising_losses = []
        l1_iter = iter(l1_dataloader)
        l2_iter = iter(l2_dataloader)

        for _ in xrange(len(l1_iter)): # Training steps
            iters += 1

            l1_sample = next(l1_iter)
            l2_sample = next(l2_iter)

            if gpuid:
                l1_sample = {k: v.cuda() for k, v in l1_sample.items()}
                l2_sample = {k: v.cuda() for k, v in l2_sample.items()}

            if iters % 2: # Denoising step

                loss_l1, output_l1 = train_step(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    src=l1_sample['src'],
                    src_lang='l1',
                    lengths=l1_sample['src_len'].data,
                    tgt=l1_sample['tgt'],
                    tgt_lang='l1')

                loss_l2, output_l2 = train_step(
                    optimizer=optimizer,
                    criterion=criterion,
                    model=model,
                    src=l2_sample['src'],
                    src_lang='l2',
                    lengths=l2_sample['src_len'].data,
                    tgt=l2_sample['tgt'],
                    tgt_lang='l2')
                denoising_losses.append((loss_l1.data[0], loss_l2.data[0]))

            else: # Backtranslation step

                pass

                # loss_l1_l2 = train_bt(
                #     optimizer=optimizer,
                #     criterion=criterion,
                #     model=model,
                #     ntokens=len(vocab_src),
                #     source=l1_sample['tgt'], # tgt because not distorted
                #     lengths=l1_sample['tgt_len'],
                #     input_language='l1',
                #     use_maxlen=False,
                #     unsup=True)

                # loss_l2_l1 = train_bt(
                #     optimizer=optimizer,
                #     criterion=criterion,
                #     model=model,
                #     ntokens=len(vocab_src),
                #     source=l2_sample['tgt'], # tgt because not distorted
                #     lengths=l2_sample['tgt_len'],
                #     input_language='l2',
                #     use_maxlen=False,
                #     unsup=True)

            if not iters % 100:
                l1_denoising_loss, l2_denoising_loss = zip(*denoising_losses)
                logging.info('Iteration: %i, L1 Denoising Loss: %0.4f' % (iters, np.mean(l1_denoising_loss)))
                logging.info('Iteration: %i, L2 Denoising Loss: %0.4f' % (iters, np.mean(l2_denoising_loss)))

            if not iters % 1000:
                logging.info('Saving model')
                torch.save(model, os.path.join(config.data.ckpt, 'model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config", required=True)
    FLAGS, _ = parser.parse_known_args()

    main(_)

