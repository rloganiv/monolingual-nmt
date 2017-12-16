import argparse
import copy
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable

from train import transform_inputs
from model_remake import Model
from utils import load_config, load_embeddings, greedy_translate
from utils import MonolingualDataset, MonolingualDataLoader


USE_CUDA = torch.cuda.is_available()


class Beam(object):
    """Ordered beam of candidate outputs.

    Code borrowed from OpenNMT PyTorch implementation:
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/Beam.py
    """
    def __init__(self, size, vocab, init_decoder, cuda=False):
        self.size = size
        self.done = False
        self.pad = vocab['<pad>']
        self.bos = vocab['<s>']
        self.eos = vocab['</s>']
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()

        # The backpointers at each time-step.
        self.prevKs = []

        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size).fill_(self.pad)]
        self.nextYs[0][0] = self.bos

        # The decoder states of the backpointers
        # Used to feed back into the decoder
        self.decStates = [init_decoder] + [None]*(self.size-1)

    # Get the outputs for the current timestep.
    def get_current_state(self):
        """Get state of beam."""
        return self.nextYs[-1]

    # Get the backpointers for the current timestep.
    def get_current_origin(self):
        """Get the backpointer to the beam at this step."""
        return self.prevKs[-1]

    # Get the decoder states for the current timestep
    def get_current_decoder_state(self):
        return self.decStates

    #  Given prob over words for every last beam `wordLk` and attention
    #   `attnOut`: Compute and update the beam search.
    #
    # Parameters:
    #
    #     * `wordLk`- probs of advancing from the last step (K x words)
    #     * `decoderOut`- decoder states after advancing (list of len K)
    #
    # Returns: True if beam selfearch is complete.
    def advance(self, workd_lk, hidden_out):
        """Advance the beam."""
        num_words = workd_lk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beam_lk = workd_lk + self.scores.unsqueeze(1).expand_as(workd_lk)
        else:
            beam_lk = workd_lk[0]

        flat_beam_lk = beam_lk.view(-1)

        bestScores, bestScoresId = flat_beam_lk.topk(self.size, 0, True, True)
        self.scores = bestScores
        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prev_k = bestScoresId / num_words
        self.prevKs.append(prev_k)
        self.nextYs.append(bestScoresId - prev_k * num_words)
        self.decStates = [hidden_out[k]for k in prev_k]
        # End condition is when top-of-beam is EOS.
        if self.nextYs[-1][0] == self.eos:
            self.done = True

        return self.done

    def sort_best(self):
        """Sort the beam."""
        return torch.sort(self.scores, 0, True)

    # Get the score of the best in the beam.
    def get_best(self):
        """Get the most likely candidate."""
        scores, ids = self.sort_best()
        return scores[1], ids[1]

    # Walk back to construct the full hypothesis.
    #
    # Parameters.
    #
    #     * `k` - the position in the beam to construct.
    #     * `to_word` - converts list of idxs into word string
    #
    # Returns.
    #
    #     1. The hypothesis
    def get_hyp(self, k, to_word=True):
        """Get hypotheses."""
        hyp = []
        # print(len(self.prevKs), len(self.nextYs), len(self.attn))
        for j in range(len(self.prevKs)-1, -1, -1):
            hyp.append(self.nextYs[j + 1][k])
            k = self.prevKs[j][k]
        return hyp[::-1] # Reverse list


def idxs_to_string(idxs, idx2word):
    """Converts list of word indices into a sentence.

    Args:
        idxs: List of word indices.
        idx2word: Maps indices to words.

    Returns:
        The sentence as a string.
    """
    bos = '<s>'
    eos = '</s>'
    words = [idx2word[idx] for idx in idxs if (idx2word[idx] != bos) and (idx2word[idx] != eos)]
    return ' '.join(words)


def translate(model, src, src_lang, lengths, beam_size, max_tgt_len, target_vocab, verbose=False):
    """Translates source sentence using beam search.

    Args:
        model - encoder/decoder model
        src - sentence from source langauge (batch size 1)
        src_lang - 'l1' or 'l2'
        lengths - length of src
        max_tgt_len - maximum length of decoded sentence
        verbose - at each decoding step, prints out highest probability sentence

    Returns:
        Translated sentence as a string.
    """
    # Use decoder and generator opposite the source langauge
    if src_lang == 'l1':
        decoder = model.l2_decoder
        generator = model.l2_to_vocab
    elif src_lang == 'l2':
        decoder = model.l1_decoder
        generator = model.l1_to_vocab
    else:
        raise ValueError('src_lang')

    enc_hidden, context = model.encoder(src, src_lang, lengths)
    dec_state = decoder.init_decoder_state(src, context, enc_hidden)

    beam = Beam(beam_size, target_vocab._word2idx, dec_state, cuda=USE_CUDA)
    idx2word = target_vocab._idx2word
    vocab_size = len(idx2word)
    logsoftmax = torch.nn.LogSoftmax() # Use log softmax since beam search code adds probabilities

    done = False
    depth = 0
    while (not done) and (depth < max_tgt_len):
        depth += 1
        next_word_lk = torch.zeros((beam_size, vocab_size)) # Likelihood of next words
        next_dec_states = []
        cur_words = beam.get_current_state() # words in beam
        cur_dec_states = beam.get_current_decoder_state() # to feed into decoder

        # Decode one step for each word in beam
        for i in range(beam_size):
            word = cur_words[i]
            dec_state = copy.copy(cur_dec_states[i])
            if word == beam.pad:
                next_dec_states.append(None)
            else:
                word = Variable(torch.LongTensor([word])).unsqueeze(0).unsqueeze(2)
                if USE_CUDA:
                    word = word.cuda()
                dec_out, dec_state, _ = decoder(word, context, dec_state,
                                                context_lengths=lengths)
                preds = generator(dec_out)
                preds_prob = logsoftmax(preds.squeeze(1))
                # Probabilities of next words
                next_word_lk[i,:] = preds_prob.data
                # Save decoder states
                next_dec_states.append(dec_state)
        done = beam.advance(next_word_lk, next_dec_states)
        if verbose: # Print top sentence at each time point
            print idxs_to_string(beam.get_hyp(0), idx2word)
    return idxs_to_string(beam.get_hyp(0), idx2word)


def main(_):
    config = load_config(FLAGS.config)

    # Load saved model
    print "Loading model"
    model_path = os.path.join(config.data.ckpt, 'model.pt')
    model = torch.load(model_path)
    model.eval()

    # Load embeddings and (test) datasets
    l1_embeddings, l1_vocab = load_embeddings(path=config.data.l1_embeddings)
    l2_embeddings, l2_vocab = load_embeddings(path=config.data.l2_embeddings)

    # Translate all test files
    start = time.time()
    beam_size = 12

    test_dirs = ['data/test_en', 'data/test_fr']

    for test_dir in test_dirs:
        src_lang = test_dir.split('_')[-1]
        if src_lang == 'en':
            src_lang = 'l1'
            src_vocab = l1_vocab
            tgt_lang = 'l2'
            tgt_vocab = l2_vocab
        elif src_lang == 'fr':
            src_lang = 'l2'
            src_vocab = l2_vocab
            tgt_lang = 'l1'
            tgt_vocab = l1_vocab
        else:
            ValueError('source language')

        test_dataset = MonolingualDataset(folder=test_dir, vocab=src_vocab)
        test_loader = MonolingualDataLoader(test_dataset)
        test_file = test_dataset._paths[0].split('/')[2]
        print test_file, src_lang
        f = open('test_translated/' + test_file + '_translated', 'w')
        for i, sample in enumerate(test_loader):
            sample = {k: v.cuda() for k, v in sample.items() if v is not None}
            src, lengths, _, _ = transform_inputs(
                src=sample['src'],
                lengths=sample['src_len'],
                tgt=sample['src'])
            translated = translate(model, src, src_lang, lengths.data, beam_size,
                                   config.data.max_length, tgt_vocab)
            f.write(translated+'\n')
        f.close()
        print("Time to translate file (secs): ",  time.time() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to json config", required=True)
    FLAGS, _ = parser.parse_known_args()

    main(_)

