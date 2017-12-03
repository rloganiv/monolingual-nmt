"""Evaluation utils."""
import sys



import torch
import torch.nn.functional as F
from torch.autograd import Variable
from data_utils import get_minibatch, get_autoencode_minibatch
from collections import Counter
import math
import numpy as np
import subprocess
import sys


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in xrange(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in xrange(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in xrange(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats


def bleu(stats):
    """Compute BLEU given n-gram statistics."""
    if len(filter(lambda x: x == 0, stats)) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)


def get_bleu(hypotheses, reference):
    """Get validation BLEU score for dev set."""
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def get_bleu_moses(hypotheses, reference):
    """Get BLEU score with moses bleu score."""
    with open('tmp_hypotheses.txt', 'w') as f:
        for hypothesis in hypotheses:
            f.write(' '.join(hypothesis) + '\n')

    with open('tmp_reference.txt', 'w') as f:
        for ref in reference:
            f.write(' '.join(ref) + '\n')

    hypothesis_pipe = '\n'.join([' '.join(hyp) for hyp in hypotheses])
    pipe = subprocess.Popen(
        ["perl", 'multi-bleu.perl', '-lc', 'tmp_reference.txt'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    pipe.stdin.write(hypothesis_pipe)
    pipe.stdin.close()
    return pipe.stdout.read()


def decode_minibatch(
    config,
    model,
    input_lines_src,
    input_lines_tgt,
    lengths,
    l1_decoder=False
):
    """Decode a minibatch."""
    for i in xrange(config['data']['max_len']):

        decoder_logit = model(input_lines_src, input_lines_tgt, lengths,l1_decoder=l1_decoder)
        word_probs = model.decode(decoder_logit, l1_decoder=l1_decoder)
        decoder_argmax = word_probs.data.cpu().numpy().argmax(axis=-1)
        next_preds = Variable(
            torch.from_numpy(decoder_argmax[:, -1])
        )

        input_lines_tgt = torch.cat(
            (input_lines_tgt, next_preds.unsqueeze(1)),
            1
        )

    return input_lines_tgt


def model_perplexity(
    model, src, src_test, tgt,
    tgt_test, config, loss_criterion,
    src_valid=None, tgt_valid=None, verbose=False,
):
    """Compute model perplexity."""
    # Get source minibatch
    losses = []
    for j in xrange(0, len(src_test['data']) // 100, config['data']['batch_size']):
        input_lines_src, output_lines_src, lens_src, mask_src = get_minibatch(
            src_test['data'], src['word2id'], j, config['data']['batch_size'],
            config['data']['max_src_length'], add_start=True, add_end=True
        )
        input_lines_src = Variable(input_lines_src.data, volatile=True)
        output_lines_src = Variable(input_lines_src.data, volatile=True)
        mask_src = Variable(mask_src.data, volatile=True)

        # Get target minibatch
        input_lines_tgt_gold, output_lines_tgt_gold, lens_src, mask_src = (
            get_minibatch(
                tgt_test['data'], tgt['word2id'], j,
                config['data']['batch_size'], config['data']['max_tgt_length'],
                add_start=True, add_end=True
            )
        )
        input_lines_tgt_gold = Variable(input_lines_tgt_gold.data, volatile=True)
        output_lines_tgt_gold = Variable(output_lines_tgt_gold.data, volatile=True)
        mask_src = Variable(mask_src.data, volatile=True)

        decoder_logit = model(input_lines_src, input_lines_tgt_gold)

        loss = loss_criterion(
            decoder_logit.contiguous().view(-1, decoder_logit.size(2)),
            output_lines_tgt_gold.view(-1)
        )

        losses.append(loss.data[0])

    return np.exp(np.mean(losses))
    
def evaluate_alignment_model(model_l1_l2, model_l2_l1, test_data, config, metric='bleu'):
    """Evaluate model."""
    
    test_iter = iter(test_data)
    preds_l1 = []
    ground_truths_l1 = []
    preds_l2=[]
    ground_truths_l2=[]
    count=0
    print ("-------Test recosntructions----------")
    for j in xrange(0, len(test_data), config['data']['batch_size']):
        
        ##################### L1 => L2 #############################
        source, target, src_flat, tgt_flat, lengths = test_iter.next()
        input_lines_target = Variable(torch.LongTensor([1]*source.size(0)).unsqueeze(1))
        
        # Decode a minibatch greedily __TODO__ add beam search decoding
        input_lines_tgt = decode_minibatch(
            config, model_l1_l2, source,
            input_lines_target, lengths[0]
        )

        # Copy minibatch outputs to cpu and convert ids to words
        input_lines_tgt = input_lines_tgt.data.cpu().numpy()
        input_lines_tgt = [
            [test_data.dictionary_tgt.idx2word[x] for x in line]
            for line in input_lines_tgt
        ]

        # Do the same for gold sentences
        output_lines_tgt_gold = target.data.cpu().numpy()
        output_lines_tgt_gold = [
            [test_data.dictionary_tgt.idx2word[x] for x in line]
            for line in output_lines_tgt_gold
        ]

        while count<5:
            logging.info('Predicted : %s ' % (' '.join(input_lines_tgt[0])))
            logging.info('-----------------------------------------------')
            logging.info('Real : %s ' % (' '.join(output_lines_tgt_gold[0])))
            logging.info('===============================================')
            count+=1
            
        # Process outputs
        for sentence_pred, sentence_real in zip(
            input_lines_tgt,
            output_lines_tgt_gold ):
            if '<eos>' in sentence_pred:
                index = sentence_pred.index('<eos>')
            else:
                index = len(sentence_pred)
            preds_l1.append(['<sos>'] + sentence_pred[:index + 1])

            if '<eos>' in sentence_real:
                index = sentence_real.index('<sos>')
            else:
                index = len(sentence_real)
            ground_truths_l1.append(['<sos>'] + sentence_real[:index + 1])
        
        
        
        ##################### L2 => L1 #############################
        
        input_lines_target = Variable(torch.LongTensor([1]*target.size(0)).unsqueeze(1))
        
        # Decode a minibatch greedily
        input_lines_tgt = decode_minibatch(
            config, model_l2_l1, target,
            input_lines_target, lengths
        )

    
        # Copy minibatch outputs to cpu and convert ids to words
        input_lines_tgt = input_lines_tgt.data.cpu().numpy()
        input_lines_tgt = [
            [test_data.dictionary_tgt.idx2word[x] for x in line]
            for line in input_lines_tgt
        ]

        # Do the same for gold sentences
        output_lines_src_gold = source.data.cpu().numpy()
        output_lines_src_gold = [
            [test_data.dictionary_tgt.idx2word[x] for x in line]
            for line in output_lines_src_gold
        ]

        # Process outputs
        for sentence_pred, sentence_real in zip(
            input_lines_tgt,
            output_lines_src_gold ):
            if '<eos>' in sentence_pred:
                index = sentence_pred.index('<eos>')
            else:
                index = len(sentence_pred)
            preds_l2.append(['<sos>'] + sentence_pred[:index + 1])

            if '<eos>' in sentence_real:
                index = sentence_real.index('<sos>')
            else:
                index = len(sentence_real)
            ground_truths_l2.append(['<sos>'] + sentence_real[:index + 1])
        
    bleu_l2_l1 = get_bleu(preds_l2, ground_truths_l2)
    bleu_l1_l2 = get_bleu(preds_l1, ground_truths_l1)
    
    return bleu_l1_l2, bleu_l2_l1

def evaluate_autoencoder_model(model, test_data, config, metric='bleu'):
    """Evaluate model."""
    preds = []
    ground_truths = []
    test_iter = iter(test_data)
    for j in xrange(0, len(test_data), config['data']['batch_size']):

        source, target, output_lines_tgt, lengths = test_iter.next()
        
        input_lines_target = Variable(torch.LongTensor([1]*source.size(0)).unsqueeze(1))
        
        # Decode a minibatch greedily __TODO__ add beam search decoding
        input_lines_tgt = decode_minibatch(
            config, model, source,
            input_lines_target,lengths
        )

        # Copy minibatch outputs to cpu and convert ids to words
        input_lines_tgt = input_lines_tgt.data.cpu().numpy()
        input_lines_tgt = [
            [test_data.dataset.dictionary_tgt.idx2word[x] for x in line]
            for line in input_lines_tgt
        ]

        # Do the same for gold sentences
        output_lines_tgt = output_lines_tgt.data.cpu().numpy()
        output_lines_tgt = [
            [test_data.dataset.dictionary_tgt.idx2word[x] for x in line]
            for line in output_lines_tgt
        ]
        # Process outputs
        for sentence_pred, sentence_real in \
                                zip(input_lines_tgt, output_lines_tgt):
            if '<eos>' in sentence_pred:
                index = sentence_pred.index('<eos>')
            else:
                index = len(sentence_pred)
            preds.append(['<sos>'] + sentence_pred[:index + 1])

            if '<eos>' in sentence_real:
                index = sentence_real.index('<eos>')
            else:
                index = len(sentence_real)
            ground_truths.append(['<sos>'] + sentence_real[:index + 1])

    return get_bleu(preds, ground_truths)


def evaluate_mono_nmt(model, test_data, config, metric='bleu'):
    """Evaluate model."""
    preds_l1 = []
    ground_truths_l1 = []
    preds_l2 = []
    ground_truths_l2 = []
    test_iter = iter(test_data)
    count=0
    for j in xrange(0, len(test_iter)):

        ##################### L1 => L2 #############################
        source, target, src_flat, tgt_flat, lengths = test_iter.next()
        input_lines_target = Variable(torch.LongTensor([1]*source.size(0)).unsqueeze(1))
        
        # Decode a minibatch greedily __TODO__ add beam search decoding
        input_lines_tgt = decode_minibatch(
            config, model, source,
            input_lines_target, lengths[0],
            l1_decoder=False
        )

        # Copy minibatch outputs to cpu and convert ids to words
        input_lines_tgt = input_lines_tgt.data.cpu().numpy()
        input_lines_tgt = [
            [test_data.dataset.dictionary_tgt.idx2word[x] for x in line]
            for line in input_lines_tgt
        ]

        # Do the same for gold sentences
        output_lines_tgt_gold = target.data.cpu().numpy()
        output_lines_tgt_gold = [
            [test_data.dataset.dictionary_tgt.idx2word[x] for x in line]
            for line in output_lines_tgt_gold
        ]

        while count<10:
            print('Predicted : %s ' % (' '.join(input_lines_tgt[0])))
            print('-----------------------------------------------')
            print('Real : %s ' % (' '.join(output_lines_tgt_gold[0])))
            print('===============================================')
            count+=1
            
        # Process outputs
        for sentence_pred, sentence_real in zip(
            input_lines_tgt,
            output_lines_tgt_gold ):
            if '<eos>' in sentence_pred:
                index = sentence_pred.index('<eos>')
            else:
                index = len(sentence_pred)
            preds_l1.append(['<sos>'] + sentence_pred[:index + 1])

            if '<eos>' in sentence_real:
                index = sentence_real.index('<eos>')
            else:
                index = len(sentence_real)
            ground_truths_l1.append(['<sos>'] + sentence_real[:index + 1])
        
        
        
        ##################### L2 => L1 #############################
        
        input_lines_target = Variable(torch.LongTensor([1]*target.size(0)).unsqueeze(1))
        
        # Decode a minibatch greedily
        input_lines_tgt = decode_minibatch(
            config, model, target,
            input_lines_target, lengths[1],
            l1_decoder=True
        )

    
        # Copy minibatch outputs to cpu and convert ids to words
        input_lines_tgt = input_lines_tgt.data.cpu().numpy()
        input_lines_tgt = [
            [test_data.dataset.dictionary_tgt.idx2word[x] for x in line]
            for line in input_lines_tgt
        ]

        # Do the same for gold sentences
        output_lines_src_gold = source.data.cpu().numpy()
        output_lines_src_gold = [
            [test_data.dataset.dictionary_tgt.idx2word[x] for x in line]
            for line in output_lines_src_gold
        ]

        # Process outputs
        for sentence_pred, sentence_real in zip(
            input_lines_tgt,
            output_lines_src_gold ):
            if '<eos>' in sentence_pred:
                index = sentence_pred.index('<eos>')
            else:
                index = len(sentence_pred)
            preds_l2.append(['<sos>'] + sentence_pred[:index + 1])

            if '<eos>' in sentence_real:
                index = sentence_real.index('<sos>')
            else:
                index = len(sentence_real)
            ground_truths_l2.append(['<sos>'] + sentence_real[:index + 1])
        
    bleu_l2_l1 = get_bleu(preds_l2, ground_truths_l2)
    bleu_l1_l2 = get_bleu(preds_l1, ground_truths_l1)
    
    return bleu_l1_l2, bleu_l2_l1
