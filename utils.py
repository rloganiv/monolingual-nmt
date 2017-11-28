'''
Created on Nov 14, 2017

@author: ddua
'''
import random
import torch.utils.data as data
import os
import math
import pickle as pkl
import torch
import numpy as np
from scipy.stats import mode
from torch.autograd import Variable

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.lvt=True
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<unk>'] = 3
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
            if not self.lvt:
                self.word2idx[word] = len(self.word2idx)
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                    {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        # add all chosen words to new vocabulary/dict
        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)
    
class NewsDataset(data.Dataset):

    def __init__(self, args, train=True, unsup=True, src=True, vocab=None):
        self.train = train
        self.src_data=[]
        self.tgt_data=[]
        self.unsupervised = unsup
        self.source = src
        if unsup:
            self.train_src_unaligned_path = os.path.join(args.data.dir, 'src')
            self.train_tgt_unaligned_path = os.path.join(args.data.dir, 'tgt')
        else:
            self.train_src_aligned_path = os.path.join(args.data.dir, 'src')
            self.train_tgt_aligned_path = os.path.join(args.data.dir, 'tgt')
            
        self.test_src_path = os.path.join(args.data.dir, 'newstest2014.en')
        self.test_tgt_path = os.path.join(args.data.dir, 'newstest2014.de')
        self.lowercase = True
        if vocab is not None:
            self.dictionary_src = vocab[0]
            self.dictionary_tgt = vocab[1]
        else:
            self.dictionary_src = Dictionary()
            self.dictionary_tgt = Dictionary()
            
        if src:
            self.vocab_size = args.data.src_vocab_size
        else:
            self.vocab_size = args.data.tgt_vocab_size
        self.args = args
        
        if self.train:
            if not unsup:
                temp_s_b=[]
                temp_s_l=[]
                self.make_vocab(self.train_src_aligned_path, vocab[0])
                self.make_vocab(self.train_tgt_aligned_path, vocab[1])
                if train:
                    for root, _, files in os.walk(self.train_src_aligned_path):
                        for file_name in files:
                            self.src_data.extend(self.tokenize(root+"/"+file_name, vocab[0]))
                            
                    for root, _, files in os.walk(self.train_tgt_aligned_path):
                        for file_name in files:
                            self.tgt_data.extend(self.tokenize(root+"/"+file_name, vocab[1]))
                            
                else:
                    self.src_data, self.tgt_data = self.tokenize([self.train_src_aligned_path, self.train_tgt_aligned_path], vocab)
                
            else:
                if src:
                    temp_s_b=[]
                    temp_s_l=[]
                    self.make_vocab(self.train_src_unaligned_path, self.dictionary_src)
                    for root, _, files in os.walk(self.train_src_unaligned_path):
                        for file_name in files:
                            a,b = self.tokenize(root+"/"+file_name, self.dictionary_src)
                            temp_s_b.extend(a)
                            temp_s_l.extend(b)
                    self.src_data.append(temp_s_b)
                    self.src_data.append(temp_s_l)
                else:
                    temp_t_b=[]
                    temp_t_l=[]        
                    self.make_vocab(self.train_tgt_unaligned_path, self.dictionary_tgt)
                    for root, _, files in os.walk(self.train_tgt_unaligned_path):
                        for file_name in files:
                            a,b = self.tokenize(root+"/"+file_name, self.dictionary_tgt)
                            temp_t_b.extend(a)
                            temp_t_l.extend(b)
                    self.tgt_data.append(temp_t_b)
                    self.tgt_data.append(temp_t_l)

        if not self.train:
            self.src_data, self.tgt_data = self.tokenize([self.test_src_path , self.test_tgt_path], vocab)
            
            assert(len(self.src_data) == len(self.tgt_data))
            
#             for s, t in zip(src_data, tgt_data):
#                 self.src_data.append(s)
#                 self.tgt_data.append(t)
                    
            
    def __getitem__(self, index):
        if self.train:
            if self.unsupervised:
                if self.source:
                    return self.src_data[index]
                else:
                    return self.tgt_data[index]
            else:
                return self.src_data[index], self.tgt_data[index]
                
        else:
            return self.test_data[index]
        
    def __len__(self):
        if self.train:
            if self.unsupervised:
                if self.source:
                    return len(self.src_data)
                else:
                    return len(self.tgt_data)
            else:
                return len(self.src_data)
        else:
            return len(self.src_data)
        
    def make_vocab(self, sentence_path, corpus_dict):
        # Add words to the dictionary
        for root, dirs, files in os.walk(sentence_path):
            for fn in files:
                with open(os.path.join(root,fn), 'r') as f:
                    for line in f:
                        if self.lowercase:
                            # -1 to get rid of \n character
                            words = line.strip().lower().split(" ")
                        else:
                            words = line.strip().split(" ")
                        for word in words:
                            #word = word.decode('Windows-1252').encode('utf-8')
                            corpus_dict.add_word(word)

        # prune the vocabulary
        corpus_dict.prune_vocab(k=self.vocab_size, cnt=False)
        pkl.dump(corpus_dict.word2idx, open(self.args.data.dir+"/vocab.pkl", 'w'))
        
    def process_line(self, line):
        try:
            if self.lowercase:
                words = line.strip().lower().split(" ")
            else:
                words = line.strip().split(" ")
        except Exception,e:
            print e
            print(line)
        return words
            
    def tokenize(self, path, corpus_dict):
        """Tokenizes a text file."""
        dropped = 0
        lines = []
        data=[]
        if type(path)==list:
            for p in path:
                data.append(open(p, 'r').read().strip().split('\n'))
            data = zip(data[0], data[1])
            vocab_s = corpus_dict[0].word2idx
            unk_idx_s = vocab_s['<unk>']
            vocab_t = corpus_dict[1].word2idx
            unk_idx_t = vocab_t['<unk>']
        else:
            data = open(path, 'r').read().split('\n')
            vocab = corpus_dict.word2idx
            unk_idx = vocab['<unk>']
        linecount = 0
        
        for line in data:
            linecount += 1
            if type(path)==list:
                words_s, words_t = self.process_line(line[0]), self.process_line(line[1])
            
                if (len(words_s) > self.args.data.max_len-1) or (len(words_t) > self.args.data.max_len-1):
                    dropped += 1
                    continue
                words_s = ['<sos>'] + words_s
                words_s += ['<eos>']
                words_t = ['<sos>'] + words_t
                words_t += ['<eos>']
                
                word_indices_s = [vocab_s[w] if w in vocab_s else unk_idx_s for w in words_s]
                word_indices_t = [vocab_t[w] if w in vocab_t else unk_idx_t for w in words_t]
                lines.append([word_indices_s, word_indices_t])
            else:
                words = self.process_line(line)
                
                words = ['<sos>'] + words
                words += ['<eos>']
                
                if (len(words) > self.args.data.max_len):
                    dropped += 1
                    continue
                # vectorize
                word_indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append(word_indices)
            
        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        
        del data
        if type(path)==list:
            lengths_src = [len(x[0])-1 for x in lines]
            lengths_tgt = [len(x[1])-1 for x in lines]
            temp=zip(*lines)
            batch, lengths = self.length_sort(zip(*((zip(*lines) + [lengths_tgt]))), lengths_src)
            batch_src, batch_tgt, batch_tgt_len = zip(*batch)
            return [batch_src, lengths] , [batch_tgt, batch_tgt_len]
        else:
            lengths = [len(x)-1 for x in lines]
            batch, lengths = self.length_sort(lines, lengths)
            return [list(batch), list(lengths)]
    
    def length_sort(self,items, lengths):
        """In order to use pytorch variable length sequence package"""
        items = list(zip(items, lengths))
        items.sort(key=lambda x: x[-1], reverse=True)
        items, lengths = zip(*items)
        return items, lengths

class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size    
        self.shuffle = shuffle
        
    def __iter__(self):
        if self.shuffle:
            temp = self.dataset.src_data if self.dataset.source else self.dataset.tgt_data
            batch = temp[0]; lengths = temp[1]
            median_length_count = mode(lengths)[0][0]
            std=len(np.where(lengths==median_length_count)[0])
            max = np.max([lengths.count(x) for x in set(lengths)])
            items = list(zip(batch, lengths))
            items.sort(key = lambda x: (x[1] *(std*100)*-1) - random.randint(1,max))
            items, lengths = zip(*items)
            if self.dataset.source:
                self.dataset.src_data = (items, lengths)
            else:
                self.dataset.tgt_data = (items, lengths)
        return DataLoaderIter(self)

    def __len__(self):
        if len(self.dataset.src_data)>0:
            return len(self.dataset.src_data)/float(self.batch_size)
        else:
            return len(self.dataset.tgt_data)/float(self.batch_size)
        
    

class DataLoaderIter(object):
    "Iterates once over the DataLoader's dataset, as specified by the sampler"

    def __init__(self, loader):
        self.dataset = loader.dataset
        if loader.dataset.src_data:
            self.dataset_len = len(loader.dataset.src_data[0])
        else:
            self.dataset_len = len(loader.dataset.tgt_data[0])
        self.counter = 0
        self.batch_size = loader.batch_size

    def __len__(self):
        return math.ceil(len(self.dataset.src_data[0])/float(self.batch_size))
     
    def collate_mono(self, batch):
        lengths = [len(x)-1 for x in batch]
        max_len = min(np.max(lengths), self.dataset.args.data.max_len)
        src_tensor = []
        targets = []
        tgt_tensor = []
        for sentence in batch:
            src = sentence[:-1]
            tgt = sentence[1:]
        
            if len(src)<max_len:
                zeros = (max_len-len(src))*[0]
                src += zeros
            if len(tgt)<max_len:
                zeros = (max_len-len(tgt))*[0]
                tgt += zeros
            
            src = src[:max_len]
            tgt = tgt[:max_len]
            src_tensor.append(torch.LongTensor(src).unsqueeze(0))
            tgt_tensor.append(torch.LongTensor(tgt).unsqueeze(0))
            targets.extend(tgt)
            
        return Variable(torch.cat(src_tensor, 0)), Variable(torch.LongTensor(targets)),\
                        Variable(torch.cat(tgt_tensor, 0)), Variable(torch.LongTensor(lengths))
    
    def collate_aligned(self, batch_src, batch_tgt):
        lengths_src = [len(x) for x in batch_src]
        lengths_tgt = [len(x) for x in batch_tgt]
        max_len_src = min(np.max(lengths_src), self.dataset.args.data.max_len)
        max_len_tgt = min(np.max(lengths_tgt), self.dataset.args.data.max_len)
        src_tensor = []
        tgt_tensor = []
        targets_flat = []
        source_flat = []
        for sample in zip(batch_src, batch_tgt):
            sentence1, sentence2 = sample
            src = sentence1[:-1]
            tgt = sentence2[1:]
            
            if len(src)<max_len_src:
                zeros = (max_len_src-len(src))*[0]
                src += zeros
                
            if len(tgt)<max_len_tgt:
                zeros = (max_len_tgt-len(tgt))*[0]
                tgt += zeros
            
            src = src[:max_len_src]
            tgt = tgt[:max_len_src]
            
            src_tensor.append(torch.LongTensor(src).unsqueeze(0))
            tgt_tensor.append(torch.LongTensor(tgt).unsqueeze(0))
            targets_flat.extend(tgt)
            source_flat.extend(src)
            
        return Variable(torch.cat(src_tensor, 0)), Variable(torch.cat(tgt_tensor, 0)), Variable(torch.LongTensor(source_flat)), \
            Variable(torch.LongTensor(targets_flat)), (Variable(torch.LongTensor(lengths_src)), Variable(torch.LongTensor(lengths_tgt)))
            
    def __next__(self):
        new_batch_length = min(self.dataset_len-self.counter, self.batch_size)
        if self.dataset.unsupervised:
            items = self.dataset.src_data
            if self.dataset.source:
                batch = self.collate_mono(self.dataset.src_data[0][self.counter:self.counter+new_batch_length])
            else:
                batch = self.collate_mono(self.dataset.tgt_data[0][self.counter:self.counter+new_batch_length])
        else:
            batch = self.collate_aligned(self.dataset.src_data[0][self.counter:self.counter+new_batch_length],\
                                         self.dataset.tgt_data[0][self.counter:self.counter+new_batch_length])
        self.counter = self.counter+new_batch_length
        return batch
    
    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var

def load_embeddings(root = '/home/ddua/data/snli/snli_lm/'):
    vocab_path=root+'vocab_41578.pkl'
    file_path=root+'embeddings'
    vocab = pkl.load(open(vocab_path))
    
    embeddings = torch.FloatTensor(len(vocab),100).uniform_(-0.1, 0.1)
    embeddings[0].fill_(0)
    embeddings[1].copy_(torch.FloatTensor(map(float,open(file_path).read().split('\n')[0].strip().split(" ")[1:])))
    embeddings[2].copy_(embeddings[1])
    
    with open(file_path) as fr:
        for line in fr:
            elements=line.strip().split(" ")
            word = elements[0]
            emb = torch.FloatTensor(map(float, elements[1:]))
            if word in vocab:
                embeddings[vocab[word]].copy_(emb)
            
    return embeddings

def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var

    
