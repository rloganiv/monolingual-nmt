"""Utilities for loading/processing data."""

import collections
import logging
import json
import os
import random
import torch

from onmt import Beam
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, SequentialSampler, BatchSampler

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


def load_embeddings(path):
    """Loads embeddings and vocabulary from an embedding file.

    Args:
        path: (string) Path to the embedding file.

    Returns:
        embeddings: (FloatTensor) vocab_size x embedding_size.
        vocab: (Vocab) Vocabulary mapping words to ids.
    """
    with open(path, 'r') as f:
        # Get dimensions from first line
        line = f.readline()
        vocab_size, embedding_size = map(int, line.strip().split())
        vocab_size += 3
        # Initialize
        embeddings = torch.FloatTensor(vocab_size, embedding_size)
        embeddings[0].copy_(torch.FloatTensor(embedding_size).uniform_(-0.1, 0.1)) # <pad>
        embeddings[1].copy_(torch.FloatTensor(embedding_size).uniform_(-0.1, 0.1)) # <unk>
        embeddings[2].copy_(torch.FloatTensor(embedding_size).uniform_(-0.1, 0.1)) # <s>
        word_list = []
        # Parse file
        for i, line in enumerate(f):
            line = line.strip().split()
            word, embedding = line[0], line[1:]
            embedding = torch.FloatTensor(map(float, embedding))
            word_list.append(word)
            embeddings[i+3].copy_(embedding)
        vocab = Vocab(word_list)
        return embeddings, vocab


def pad_and_collate(batch):
    """Collates batches of data, where lists are padded to be the same length.

    Args:
        batch: A batch of data to be collated.

    Returns:
        The collated data.
    """
    if isinstance(batch[0], list):
        out = None
        lengths = [len(x) for x in batch]
        max_length = max(lengths)
        padded = [torch.LongTensor(x + [0]*(max_length - len(x))) for x in batch]
        return Variable(torch.stack(padded, dim=0, out=out))
    elif isinstance(batch[0], int):
        return Variable(torch.LongTensor(batch))
    elif isinstance(batch[0], collections.Mapping):
        return {key: pad_and_collate([d[key] for d in batch]) for key in batch[0]}
    error_msg = 'batch contains unexpected type %s'
    raise TypeError(error_msg % type(batch[0]))


def greedy_translate(model, src, src_lang, lengths, tgt_lang, max_length):

    # Useful.
    batch_size = src.size()[1]

    # Use correct decoder and generator.
    if tgt_lang == 'l1':
        decoder = model.l1_decoder
        generator = model.l1_to_vocab
    elif tgt_lang == 'l2':
        decoder = model.l2_decoder
        generator = model.l2_to_vocab
    else:
        raise ValueError('tgt_lang')

    # Get encodings of source sentences.
    enc_hidden, context = model.encoder(src, src_lang, lengths)
    dec_state = decoder.init_decoder_state(src, context, enc_hidden)

    # Greedy decoding
    output_lengths = torch.ones(batch_size).long()
    input = torch.LongTensor(1, batch_size, 1).fill_(2) # Fill with <s>
    all_true = torch.ones(batch_size).byte()
    if USE_CUDA:
        output_lengths = output_lengths.cuda()
        input = input.cuda()
        all_true = all_true.cuda()
    output = [input]
    input = Variable(input, volatile=True)

    for i in range(max_length):

        # Run one step
        dec_out, dec_state, attn = decoder(input, context, dec_state,
                                           context_lengths=lengths)
        _, preds = generator(dec_out).data.topk(1)


        # If prev token is <pad> or </s> predict <pad>
        terminated = input.squeeze().data.eq(0)
        terminated += input.squeeze().data.eq(3)
        preds[terminated] = 0

        # If predicted token is not <pad> add one to output_length
        is_pad = preds.squeeze().eq(0)
        output_lengths[1 - is_pad] += 1

        # Update input and output
        input = Variable(preds, volatile=True)
        output.append(preds)

        # Stopping condition - new input is all </s> or <pad>
        if torch.equal(terminated, all_true):
            break

    # Terminate any persisting sentences
    terminated = input.squeeze().data.eq(0)
    terminated += input.squeeze().data.eq(3)
    if not torch.equal(terminated, all_true):
        preds = torch.LongTensor(1, batch_size, 1).fill_(3)
        if USE_CUDA:
            preds = preds.cuda()
        preds[terminated] = 0
        output.append(preds)

    output = Variable(torch.cat(output, dim=0))
    if USE_CUDA:
        output = output.cuda()
    return output, output_lengths


class Vocab(object):
    """Streamlined implementation of vocab object.

    Args:
        word_list: A list of words in the vocab. Words assumed to be unique.
    """

    def __init__(self, word_list):
        self._idx2word = ['<pad>', '<unk>', '<s>']
        self._idx2word.extend(word_list)
        self._word2idx = {w: i for i, w in enumerate(self._idx2word)}

    def __len__(self):
        return len(self._idx2word)

    def word2idx(self, word):
        """Gets index for a given word.

        Args:
            word: (string) Word to lookup index for.
        """
        if word in self._word2idx:
            return self._word2idx[word]
        else:
            return self._word2idx['<unk>']

    def idx2word(self, idx):
        """Gets word for a given index.

        Args:
            idx: (int) Index to lookup word for.
        """
        return self._idx2word[idx]


class MonolingualDataset(Dataset):
    """Interface for a monolingual corpus split over multiple files. To be used
    in training/evaluating unsupervised neural machine translation systems.

    Args:
        folder: Path to folder containing the preprocessed monolingual corpus.
        vocab: A Vocab object used to map words to integer ids.
        eos_token: End-of-sentence token.
        train: Whether or not the dataset is used for training.
    """

    def __init__(self, folder, vocab, eos_token='</s>', sos_token='<s>', train=False):
        self._paths = [os.path.join(folder, x) for x in os.listdir(folder)]
        self._line_counts = [self._line_count(path) for path in self._paths]
        self._vocab = vocab
        self._eos_token = eos_token
        self._sos_token = sos_token
        self._train = train
        self._active_file_index = None
        self._line_cache = None

    def _line_count(self, path):
        """Gets the line count for a file.

        Args:
            path: Path of the file.

        Returns:
            int: The number of lines in the file.
        """
        with open(path, 'r') as f:
            for i, _ in enumerate(f): pass
        return i + 1

    @property
    def line_counts(self):
        return list(self._line_counts) # Copy

    def _lookup_file_index(self, line_index):
        """Looks up the index of the file containing the i'th line in the
        dataset.

        Args:
            line_index: The index of the line to look up.

        Returns:
            int: The index of the corresponding file.
        """
        total = 0
        for i, line_count in enumerate(self._line_counts):
            total += line_count
            if line_index < total:
                return i
        return IndexError('issue looking up file index')

    def _distort(self, word_ids):
        """Applies n/2 pairwise swaps to the input.

        Args:
            word_ids: A list of word_ids.

        Returns:
            list: A distorted list of word ids.
        """
        n = len(word_ids)
        out = list(word_ids) # Copy
        swap_indices = list(range(1, n-2))
        random.shuffle(swap_indices)
        swap_indices = swap_indices[:n//2]
        for ind in swap_indices:
            x, y = out[ind:ind+2]
            out[ind:ind+2] = y, x
        return out

    def __len__(self):
        return sum(self._line_counts)

    def __getitem__(self, index):
        # Check if file corresponds to one whose lines are currently cached.
        file_index = self._lookup_file_index(index)
        if file_index != self._active_file_index:
            # If not then load that file's lines into cache.
            logging.info('Loading file at index: %i' % file_index)
            self._active_file_index = file_index
            with open(self._paths[file_index], 'r') as f:
                self._line_cache = f.readlines()
        # Convert global index to within file line index and retrieve data.
        index -= sum(self._line_counts[:file_index]) + 1
        line = self._line_cache[index]
        words = [self._sos_token] + line.strip().split() + [self._eos_token]
        word_ids = [self._vocab.word2idx(word) for word in words]
        # Prepare output
        if self._train:
            src, tgt = self._distort(word_ids), word_ids
        else:
            src, tgt = word_ids, None
        out = {
            'src': src,
            'src_len': len(src),
            'tgt': tgt,
            'tgt_len': len(tgt) if tgt != None else 0
        }
        return out


class MonolingualRandomSampler(Sampler):
    """Samples elements semi-randomly, without replacement.

    Sampling ensures that subsequent samples are almost always drawn
    randomly from the same file.

    Arguments:
        data_source: A MonolingualDataset object to sample from.
    """

    def __init__(self, data_source):
        self._data_source = data_source

    def __iter__(self):
        total = 0
        out = torch.LongTensor()
        for line_count in self._data_source.line_counts:
            tmp = torch.randperm(line_count).long() + total
            out = torch.cat([out, tmp])
            total += line_count
        return iter(out)

    def __len__(self):
        return len(self._data_source)


class MonolingualDataLoader(DataLoader):
    """Data Loader which uses the proper Sampler and collate_function for a
    MonolingualDataset."""

    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0,
                 pin_memory=False, drop_last=False, timeout=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.collate_fn = pad_and_collate
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.timeout = timeout

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        if shuffle:
            sampler = MonolingualRandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.sampler = sampler
        self.batch_sampler = batch_sampler



