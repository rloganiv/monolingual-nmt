
from numpy.random import sample
import sys
from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string
from model import Seq2SeqAttention
from evaluate import evaluate_autoencoder_model, evaluate_alignment_model
import json
import math
import numpy as np
import logging
import argparse
import os
from utils import NewsDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    help="path to json config",
    required=True
)
config_file_path = parser.parse_args().config
config = read_config(config_file_path)
args = json.loads(open(config_file_path).read(), object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
experiment_name = hyperparam_string(config)
save_dir = config['data']['save_dir']
load_dir = config['data']['load_dir']
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='log/%s' % (experiment_name),
    filemode='w'
)

# define a new Handler to log to console as well
console = logging.StreamHandler()
# optional, set the logging level
console.setLevel(logging.INFO)
# set a format which is the same for console use
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)


logging.info('Model Parameters : ')
logging.info('Task : %s ' % (config['data']['task']))
logging.info('Model : %s ' % (config['model']['seq2seq']))
logging.info('Source Language : %s ' % (config['model']['src_lang']))
logging.info('Target Language : %s ' % (config['model']['trg_lang']))
logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Target Word Embedding Dim  : %s' % (config['model']['dim_word_trg']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (1))
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['model']['n_layers_trg']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

#weight_mask = torch.ones(trg_vocab_size).cuda()
#weight_mask[trg['word2id']['<pad>']] = 0
criterion_l1 = nn.CrossEntropyLoss()
criterion_l2 = nn.CrossEntropyLoss()
criterion_l1_l2 = nn.CrossEntropyLoss()
criterion_l2_l1 = nn.CrossEntropyLoss()

print 'Reading data ...'


corpus_train_src = NewsDataset(args, train=True,  unsup=True, src=True)
corpus_train_tgt = NewsDataset(args, train=True,  unsup=True, src=False)
corpus_train_aligned = NewsDataset(args, train=True,  unsup=False,\
                        vocab=[corpus_train_src.dictionary_src, \
                               corpus_train_tgt.dictionary_tgt])

corpus_test = NewsDataset(args, train=False, unsup=False,\
                        vocab=[corpus_train_src.dictionary_src, \
                               corpus_train_tgt.dictionary_tgt])

src_ntokens =  len(corpus_train_src.dictionary_src.word2idx)
tgt_ntokens =  len(corpus_train_tgt.dictionary_tgt.word2idx)

logging.info('Found %d words in src ' % (src_ntokens))
logging.info('Found %d words in trg ' % (tgt_ntokens))

src_ntokens= min(src_ntokens, args.data.src_vocab_size )
tgt_ntokens = min(tgt_ntokens, args.data.tgt_vocab_size )

train_data_l1 = DataLoader(corpus_train_src, batch_size=args.data.batch_size, shuffle=True)

train_data_l2 = DataLoader(corpus_train_tgt, batch_size=args.data.batch_size, shuffle=True)

train_aligned = DataLoader(corpus_train_aligned, batch_size=args.data.batch_size, shuffle=True)

test_data = DataLoader(corpus_test, batch_size=args.data.batch_size, shuffle=True)

gpuid = args.training.gpuid.strip().split(" ")
gpuid = map(int,gpuid) if str(gpuid[0]) else None
if len(gpuid)>1:
    autoenc_L1 = Seq2SeqAttention(
        src_emb_dim=config['model']['dim_word_src'],
        trg_emb_dim=config['model']['dim_word_src'],
        src_vocab_size=src_ntokens,
        trg_vocab_size=src_ntokens,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        ctx_hidden_dim=config['model']['dim'],
        attention_mode='dot',
        batch_size=config['data']['batch_size'],
        bidirectional=config['model']['bidirectional'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.3, gpu=(gpuid is not None)
    )
    autoencoder_L1 = torch.nn.DataParallel(autoenc_L1.cuda(), device_ids=gpuid)
    autoenc_L2 = Seq2SeqAttention(
        src_emb_dim=config['model']['dim_word_trg'],
        trg_emb_dim=config['model']['dim_word_trg'],
        src_vocab_size=src_ntokens,
        trg_vocab_size=src_ntokens,
        src_hidden_dim=config['model']['dim'],
        trg_hidden_dim=config['model']['dim'],
        ctx_hidden_dim=config['model']['dim'],
        attention_mode='dot',
        batch_size=config['data']['batch_size'],
        bidirectional=config['model']['bidirectional'],
        nlayers=config['model']['n_layers_src'],
        nlayers_trg=config['model']['n_layers_trg'],
        dropout=0.3, gpu=(gpuid is not None)
    )
    autoencoder_L2 = torch.nn.DataParallel(autoenc_L2.cuda(), device_ids=gpuid)
else:
    autoencoder_L1 = Seq2SeqAttention(
                src_emb_dim=config['model']['dim_word_src'],
                trg_emb_dim=config['model']['dim_word_src'],
                src_vocab_size=src_ntokens,
                trg_vocab_size=src_ntokens,
                src_hidden_dim=config['model']['dim'],
                trg_hidden_dim=config['model']['dim'],
                ctx_hidden_dim=config['model']['dim'],
                attention_mode='dot',
                batch_size=config['data']['batch_size'],
                bidirectional=config['model']['bidirectional'],
                nlayers=config['model']['n_layers_src'],
                nlayers_trg=config['model']['n_layers_trg'],
                dropout=0.3, gpu=(gpuid is not None))
    autoencoder_L2 = Seq2SeqAttention(
                src_emb_dim=config['model']['dim_word_trg'],
                trg_emb_dim=config['model']['dim_word_trg'],
                src_vocab_size=tgt_ntokens,
                trg_vocab_size=tgt_ntokens,
                src_hidden_dim=config['model']['dim'],
                trg_hidden_dim=config['model']['dim'],
                ctx_hidden_dim=config['model']['dim'],
                attention_mode='dot',
                batch_size=config['data']['batch_size'],
                bidirectional=config['model']['bidirectional'],
                nlayers=config['model']['n_layers_src'],
                nlayers_trg=config['model']['n_layers_trg'],
                dropout=0.3, gpu=(gpuid is not None))
    seq2seq_L1_L2 = Seq2SeqAttention(
                src_emb_dim=config['model']['dim_word_src'],
                trg_emb_dim=config['model']['dim_word_trg'],
                src_vocab_size=tgt_ntokens,
                trg_vocab_size=tgt_ntokens,
                src_hidden_dim=config['model']['dim'],
                trg_hidden_dim=config['model']['dim'],
                ctx_hidden_dim=config['model']['dim'],
                attention_mode='dot',
                batch_size=config['data']['batch_size'],
                bidirectional=config['model']['bidirectional'],
                nlayers=config['model']['n_layers_src'],
                nlayers_trg=config['model']['n_layers_trg'],
                dropout=0.3, gpu=(gpuid is not None))
    seq2seq_L2_L1 = Seq2SeqAttention(
                src_emb_dim=config['model']['dim_word_trg'],
                trg_emb_dim=config['model']['dim_word_src'],
                src_vocab_size=tgt_ntokens,
                trg_vocab_size=tgt_ntokens,
                src_hidden_dim=config['model']['dim'],
                trg_hidden_dim=config['model']['dim'],
                ctx_hidden_dim=config['model']['dim'],
                attention_mode='dot',
                batch_size=config['data']['batch_size'],
                bidirectional=config['model']['bidirectional'],
                nlayers=config['model']['n_layers_src'],
                nlayers_trg=config['model']['n_layers_trg'],
                dropout=0.3, gpu=(gpuid is not None))
    
    if len(gpuid) == 1:
        autoencoder_L1 = autoencoder_L1.cuda()
        autoencoder_L2 = autoencoder_L2.cuda()
        seq2seq_L1_L2 = seq2seq_L1_L2.cuda()
        seq2seq_L2_L1 = seq2seq_L2_L1.cuda()
    
optimizer_auto_L1 = optim.Adam(autoencoder_L1.parameters(),
                           lr=0.0002,
                           betas=(0.5, 0.999))
optimizer_auto_L2 = optim.Adam(autoencoder_L2.parameters(),
                           lr=0.0002,
                           betas=(0.5, 0.999))
optimizer_L1_L2 = optim.Adam(seq2seq_L1_L2.parameters(),
                           lr=0.0002,
                           betas=(0.5, 0.999))
optimizer_L2_L1 = optim.Adam(seq2seq_L2_L1.parameters(),
                           lr=0.0002,
                           betas=(0.5, 0.999))

if load_dir:
    autoencoder_L1.load_state_dict(torch.load(open(load_dir+"/autoencoder_L1.pt")))
    autoencoder_L2.load_state_dict(torch.load(open(load_dir+"/autoencoder_L2.pt")))

def train_ae(optimizer, criterion, autoencoder, ntokens, source, target, lengths):
    optimizer.zero_grad()
    output = autoencoder(source, target , lengths)
    flattened_output = output.view(-1, ntokens)
    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)

    masked_output = \
        flattened_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion(masked_output, masked_target)
    loss.backward()
    autoencoder.embedding.zero_grad()
    optimizer.step()
    return loss, output

def save_model(epoch=None):
    if epoch:
        torch.save(autoencoder_L1.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '__epoch_'+ str(epoch) + '_autoenc_l1.model'), 'wb'))
        torch.save(autoencoder_L2.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '__epoch_'+ str(epoch) + '_autoenc_l2.model'), 'wb'))
        torch.save(seq2seq_L1_L2.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '__epoch_'+ str(epoch) + '_seq2seq_l1_l2.model'), 'wb'))
        torch.save(seq2seq_L2_L1.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '__epoch_'+ str(epoch) + '_seq2seq_l2_l1.model'), 'wb'))
     
    else:
        torch.save(autoencoder_L1.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '_autoenc_l1.model'), 'wb'))
        torch.save(autoencoder_L2.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '_autoenc_l2.model'), 'wb'))
        torch.save(seq2seq_L1_L2.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '_seq2seq_l1_l2.model'), 'wb'))
        torch.save(seq2seq_L2_L1.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '_seq2seq_l2_l1.model'), 'wb'))
      
        
for i in xrange(1000):
    losses = []
    L2_train_iter = iter(train_data_l2)
    L1_train_iter = iter(train_data_l1)
    L1_L2_train_iter = iter(train_aligned)
        
    for j in xrange(0, len(L1_train_iter)):

        source_l1, target_l1, output_lines_trg_l1, lengths_l1 = L1_train_iter.next()
        source_l2, target_l2, output_lines_trg_l2, lengths_l2 = L2_train_iter.next()
        source_l1_aligned, target_l2_aligned, source_l1_flat, target_l2_flat, lengths = L1_L2_train_iter.next()
        
        if gpuid:
            source_l1 = source_l1.cuda()
            source_l2 = source_l2.cuda()
            target_l1 = target_l1.cuda()
            target_l2 = target_l2.cuda()
            
        loss_l1, output_l1 = train_ae(optimizer_auto_L1, criterion_l1, autoencoder_L1, src_ntokens, source_l1, target_l1, lengths_l1) 
        
        loss_l2, output_l2 = train_ae(optimizer_auto_L2, criterion_l2, autoencoder_L2, tgt_ntokens, source_l2, target_l2, lengths_l2)
        
        loss_l1_l2, output_l1_l2 = train_ae(optimizer_L1_L2, criterion_l1_l2, seq2seq_L1_L2, tgt_ntokens, source_l2, target_l2, lengths_l2)
        
        loss_l2_l1, output_l2_l1 = train_ae(optimizer_L2_L1, criterion_l2_l1, seq2seq_L2_L1, tgt_ntokens, source_l2, target_l2, lengths_l2)
        
        losses.append((loss_l1.data[0], loss_l2.data[0], loss_l1_l2.data[0], loss_l2_l1.data[0]))
        
        if (j % config['management']['print_samples'] == 0):
            l1_mean, l2_mean, l1_l2_mean, l2_l1_mean = zip(losses)
            print("L1 : {0} L2 : {1} L3 : {2} L4 : {3}".format(np.mean(l1_mean), np.mean(l2_mean), np.mean(l1_l2_mean), np.mean(l2_l1_mean)))
            word_probs_l1 = autoencoder_L1.decode(output_l1).data.cpu().numpy().argmax(axis=-1)
            
            word_probs_l2 = autoencoder_L2.decode(output_l2).data.cpu().numpy().argmax(axis=-1)

            output_lines_trg_l1 = output_lines_trg_l1.data.cpu().numpy()
            output_lines_trg_l2 = output_lines_trg_l2.data.cpu().numpy()
            
            print ("-------Training recosntructions----------")
            for sentence_pred, sentence_real in zip(
                word_probs_l1[:5], output_lines_trg_l1[:5]):
                sentence_pred = [corpus_train_src.dictionary_src.idx2word[x] for x in sentence_pred]
                sentence_real = [corpus_train_src.dictionary_src.idx2word[x] for x in sentence_real]

                if '<eos>' in sentence_real:
                    index = sentence_real.index('<eos>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('Predicted : %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real : %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')
                
            for sentence_pred, sentence_real in zip(
                word_probs_l2[:5], output_lines_trg_l2[:5]):
                sentence_pred = [corpus_train_tgt.dictionary_tgt.idx2word[x] for x in sentence_pred]
                sentence_real = [corpus_train_tgt.dictionary_tgt.idx2word[x] for x in sentence_real]

                if '<eos>' in sentence_real:
                    index = sentence_real.index('<eos>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('Predicted : %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real : %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')
                
            word_probs = seq2seq_L1_L2.decode(output_l1_l2).data.cpu().numpy().argmax(axis=-1)
            for sentence_pred, sentence_real in zip(
                word_probs[:5], target[:5]):
                sentence_pred = [corpus_train_aligned.dictionary_src.idx2word[x] for x in sentence_pred]
                sentence_real = [corpus_train_aligned.dictionary_src.idx2word[x] for x in sentence_real]

                if '<eos>' in sentence_real:
                    index = sentence_real.index('<eos>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('Predicted : %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real : %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')
                
            word_probs = seq2seq_L2_L1.decode(output_l2_l1).data.cpu().numpy().argmax(axis=-1)
            for sentence_pred, sentence_real in zip(
                word_probs[:5], source[:5]):
                sentence_pred = [corpus_train_aligned.dictionary_tgt.idx2word[x] for x in sentence_pred]
                sentence_real = [corpus_train_aligned.dictionary_tgt.idx2word[x] for x in sentence_real]

                if '<eos>' in sentence_real:
                    index = sentence_real.index('<eos>')
                    sentence_real = sentence_real[:index]
                    sentence_pred = sentence_pred[:index]

                logging.info('Predicted : %s ' % (' '.join(sentence_pred)))
                logging.info('-----------------------------------------------')
                logging.info('Real : %s ' % (' '.join(sentence_real)))
                logging.info('===============================================')

        if j % config['management']['checkpoint_freq'] == 0:

            logging.info('Evaluating model ...')
            bleu_l1 = evaluate_autoencoder_model(
                autoencoder_L1, test_data, config,
                metric='bleu',
            )
            
            bleu_l2 = evaluate_autoencoder_model(
                autoencoder_L2, test_data, config,
                metric='bleu',
            )
            
            bleu_l1_l2, bleu_l2_l1 = evaluate_autoencoder_model(
                seq2seq_L1_L2, seq2seq_L2_L1, test_data, config,
                metric='bleu',
            )
            


            logging.info('Epoch : %d Minibatch : %d : BLEU on L1 : %.5f ' % (i, j, bleu_l1))
            logging.info('Epoch : %d Minibatch : %d : BLEU on L2 : %.5f ' % (i, j, bleu_l2))
            logging.info('Epoch : %d Minibatch : %d : BLEU on L1 to L2 : %.5f ' % (i, j, bleu_l1_l2))
            logging.info('Epoch : %d Minibatch : %d : BLEU on L2 to L1 : %.5f ' % (i, j, bleu_l2_l1))

            logging.info('Saving model ...')

            save_model(i)

    bleu_l1 = evaluate_autoencoder_model(autoencoder_L1, test_data, config, metric='bleu')
    bleu_l2 = evaluate_autoencoder_model(autoencoder_L2, test_data, config, metric='bleu')

    logging.info('Epoch : %d : BLEU : %.5f ' % (i, bleu_l1))
    logging.info('Epoch : %d : BLEU : %.5f ' % (i, bleu_l2))

    save_model()
