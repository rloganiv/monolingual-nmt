
from numpy.random import sample
import sys
from data_utils import read_nmt_data, get_minibatch, read_config, hyperparam_string
from model import Seq2SeqMono
from evaluate import evaluate_autoencoder_model, evaluate_alignment_model,evaluate_mono_nmt
import json
import math
import numpy as np
import logging
import argparse
import os
from utils import NewsDataset, DataLoader, load_embeddings
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
logging.info('Target Language : %s ' % (config['model']['tgt_lang']))
logging.info('Source Word Embedding Dim  : %s' % (config['model']['dim_word_src']))
logging.info('Target Word Embedding Dim  : %s' % (config['model']['dim_word_tgt']))
logging.info('Source RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Target RNN Hidden Dim  : %s' % (config['model']['dim']))
logging.info('Source RNN Depth : %d ' % (config['model']['n_layers_src']))
logging.info('Target RNN Depth : %d ' % (1))
logging.info('Source RNN Bidirectional  : %s' % (config['model']['bidirectional']))
logging.info('Batch Size : %d ' % (config['model']['n_layers_tgt']))
logging.info('Optimizer : %s ' % (config['training']['optimizer']))
logging.info('Learning Rate : %f ' % (config['training']['lrate']))

#weight_mask = torch.ones(tgt_vocab_size).cuda()
#weight_mask[tgt['word2id']['<pad>']] = 0
criterion = nn.CrossEntropyLoss()

print 'Reading data ...'

embeddings_src, vocab_src = load_embeddings(config['data']['src_emb'])
embeddings_tgt, vocab_tgt = load_embeddings(config['data']['tgt_emb'])

src_ntoken = len(vocab_src)
tgt_ntoken = len(vocab_tgt)

corpus_train_src = NewsDataset(args, train=True,  unsup=True, src=True, vocab=[vocab_src, vocab_tgt])
corpus_train_tgt = NewsDataset(args, train=True,  unsup=True, src=False, vocab=[vocab_src, vocab_tgt])
corpus_train_aligned = NewsDataset(args, train=True,  unsup=False,\
                        vocab=[embeddings_src, embeddings_tgt])
# corpus_train_src.dictionary_src.prune_vocab(k=config['data']['src_vocab_size'], cnt=False)
# corpus_train_tgt.dictionary_tgt.prune_vocab(k=config['data']['tgt_vocab_size'], cnt=False)

corpus_test = NewsDataset(args, train=False, unsup=False,\
                        vocab=[embeddings_src, embeddings_tgt])

train_data_l1 = DataLoader(corpus_train_src, batch_size=args.data.batch_size, shuffle=True)

train_data_l2 = DataLoader(corpus_train_tgt, batch_size=args.data.batch_size, shuffle=True)

train_aligned = DataLoader(corpus_train_aligned, batch_size=args.data.batch_size, shuffle=True)

test_data = DataLoader(corpus_test, batch_size=args.data.batch_size, shuffle=True)

gpuid = args.training.gpuid.strip().split(" ")
gpuid = map(int,gpuid) if str(gpuid[0]) else None
if gpuid and len(gpuid)>1:
    mod = Seq2SeqMono(
        src_emb_dim=config['model']['dim_word_src'],
        tgt_emb_dim=config['model']['dim_word_src'],
        src_vocab_size=src_ntokens+tgt_ntokens-4,
        tgt_vocab_size_l1=src_ntokens,
        tgt_vocab_size_l2=tgt_ntokens,
        src_hidden_dim=config['model']['dim'],
        tgt_hidden_dim=config['model']['dim'],
        ctx_hidden_dim=config['model']['dim'],
        attention_mode='dot',
        batch_size=config['data']['batch_size'],
        bidirectional=config['model']['bidirectional'],
        nlayers=config['model']['n_layers_src'],
        nlayers_tgt=config['model']['n_layers_tgt'],
        maxlen = config['data']['max_len']-1,
        dropout=0.3, gpu=(gpuid is not None)
    )
    model = torch.nn.DataParallel(mod.cuda(), device_ids=gpuid)
    
else:
    model = Seq2SeqMono(
                src_emb_dim=config['model']['dim_word_src'],
                tgt_emb_dim=config['model']['dim_word_src'],
                src_vocab_size=src_ntokens+tgt_ntokens-4,
                tgt_vocab_size_l1=src_ntokens,
                tgt_vocab_size_l2=tgt_ntokens,
                src_hidden_dim=config['model']['dim'],
                tgt_hidden_dim=config['model']['dim'],
                ctx_hidden_dim=config['model']['dim'],
                attention_mode='dot',
                batch_size=config['data']['batch_size'],
                bidirectional=config['model']['bidirectional'],
                nlayers=config['model']['n_layers_src'],
                nlayers_tgt=config['model']['n_layers_tgt'],
                maxlen = config['data']['max_len']-1,
                dropout=0.3, gpu=(gpuid is not None))
    
    if gpuid and len(gpuid) == 1:
        model = model.cuda()
        
    
optimizer = optim.Adam(model.parameters(),
                           lr=0.0002,
                           betas=(0.5, 0.999))
if load_dir:
    model.load_state_dict(torch.load(open(load_dir+"/model.pt")))
    
def train_ae(optimizer, criterion, model, ntokens, source, \
             target, lengths, l1_decoder, use_maxlen, unsup):
    optimizer.zero_grad()
    output = model(source, target , lengths, l1_decoder=l1_decoder, \
                   use_maxlen=use_maxlen, unsup=unsup)
    flattened_output = output.view(-1, ntokens)
    if not unsup:
        target = target.view(-1)
    mask = target.gt(0)
    masked_target = target.masked_select(mask)
    output_mask = mask.unsqueeze(1).expand(mask.size(0), ntokens)
    masked_output = \
        flattened_output.masked_select(output_mask).view(-1, ntokens)
    loss = criterion(masked_output, masked_target)
    loss.backward()
    optimizer.step()
    return loss, output

def save_model(epoch=None):
    if epoch:
        torch.save(model.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '__epoch_'+ str(epoch) + '_model.pt'), 'wb'))
        
    else:
        torch.save(model.state_dict(),
                    open(os.path.join(save_dir,experiment_name + '_model.pt'), 'wb'))

        
for i in xrange(1000):
    losses = []
    L2_train_iter = iter(train_data_l2)
    L1_train_iter = iter(train_data_l1)
    L1_L2_train_iter = iter(train_aligned)
        
    for j in xrange(0, len(L1_train_iter)):

        source_l1, target_l1, output_lines_tgt_l1, lengths_l1 = L1_train_iter.next()
        source_l2, target_l2, output_lines_tgt_l2, lengths_l2 = L2_train_iter.next()
        source_l1_aligned, target_l2_aligned, source_l1_flat, target_l2_flat, lengths = L1_L2_train_iter.next()
        
        if gpuid:
            source_l1 = source_l1.cuda()
            source_l2 = source_l2.cuda()
            target_l1 = target_l1.cuda()
            target_l2 = target_l2.cuda()
            source_l1_aligned = source_l1_aligned.cuda()
            target_l2_aligned = target_l2_aligned.cuda()
            target_l2_flat = target_l2_flat.cuda()
            source_l1_flat = source_l1_flat.cuda()
            
        
        loss_l1, output_l1 = train_ae(optimizer, criterion, model, src_ntokens, source_l1, target_l1, lengths_l1, l1_decoder=True, use_maxlen=False, unsup=True) 
        
        loss_l2, output_l2 = train_ae(optimizer, criterion, model, tgt_ntokens, source_l2, target_l2, lengths_l2, l1_decoder=False, use_maxlen=False, unsup=True)
        
        loss_l1_l2, output_l1_l2 = train_ae(optimizer, criterion, model, tgt_ntokens_align, source_l1_aligned, target_l2_aligned, lengths[0], l1_decoder=False, use_maxlen=True, unsup=False)
        
        loss_l2_l1, output_l2_l1 = train_ae(optimizer, criterion, model, src_ntokens_align, target_l2_aligned, source_l1_aligned, lengths[1], l1_decoder=True, use_maxlen=True, unsup=False)
        
        losses.append((loss_l1.data[0], loss_l2.data[0], loss_l1_l2.data[0], loss_l2_l1.data[0]))
        
        if (j % config['management']['print_samples'] == 0):
            l1_mean, l2_mean, l1_l2_mean, l2_l1_mean = zip(*losses)
            print("L1 : {0} L2 : {1} L3 : {2} L4 : {3}".format(np.mean(l1_mean), np.mean(l2_mean), np.mean(l1_l2_mean), np.mean(l2_l1_mean)))
            word_probs_l1 = model.decode(output_l1, l1_decoder=True).data.cpu().numpy().argmax(axis=-1)
            
            word_probs_l2 = model.decode(output_l2, l1_decoder=False).data.cpu().numpy().argmax(axis=-1)

            output_lines_tgt_l1 = output_lines_tgt_l1.data.cpu().numpy()
            output_lines_tgt_l2 = output_lines_tgt_l2.data.cpu().numpy()
            
            print ("-------Training reconstructions----------")
            for sentence_pred, sentence_real in zip(
                word_probs_l1[:5], output_lines_tgt_l1[:5]):
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
                word_probs_l2[:5], output_lines_tgt_l2[:5]):
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
                
            word_probs = model.decode(output_l1_l2, l1_decoder=False).data.cpu().numpy().argmax(axis=-1)
            for sentence_pred, sentence_real in zip(
                word_probs[:5], target_l2_aligned.data.cpu()[:5]):
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
                
            word_probs = model.decode(output_l2_l1, l1_decoder=True).data.cpu().numpy().argmax(axis=-1)
            for sentence_pred, sentence_real in zip(
                word_probs[:5], source_l1_aligned.data.cpu()[:5]):
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

        if j % config['management']['checkpoint_freq'] == 0:

            logging.info('Evaluating model ...')
            bleu_l1, blue_l2,bleu_l1_l2, bleu_l2_l1  = evaluate_mono_nmt(
                model, test_data, config,
                metric='bleu',
            )

            logging.info('Epoch : %d Minibatch : %d : BLEU on L1 : %.5f ' % (i, j, bleu_l1))
            logging.info('Epoch : %d Minibatch : %d : BLEU on L2 : %.5f ' % (i, j, bleu_l2))
            logging.info('Epoch : %d Minibatch : %d : BLEU on L1 to L2 : %.5f ' % (i, j, bleu_l1_l2))
            logging.info('Epoch : %d Minibatch : %d : BLEU on L2 to L1 : %.5f ' % (i, j, bleu_l2_l1))

            logging.info('Saving model ...')

            save_model(i)

    bleu_l1, blue_l2,bleu_l1_l2, bleu_l2_l1  = evaluate_mono_nmt(
                model, test_data, config,
                metric='bleu',
            )
    logging.info('Epoch : %d : BLEU : %.5f ' % (i, bleu_l1))
    logging.info('Epoch : %d : BLEU : %.5f ' % (i, bleu_l2))
    logging.info('Epoch : %d : BLEU on L1 to L2 : %.5f ' % (i, bleu_l1_l2))
    logging.info('Epoch : %d : %d : BLEU on L2 to L1 : %.5f ' % (i, bleu_l2_l1))


    save_model()
