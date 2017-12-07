"""Sequence to Sequence models."""
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class StackedAttentionGRU(nn.Module):
    """Deep Attention GRU."""

    def __init__(
        self,
        input_size,
        rnn_size,
        num_layers,
        batch_first=True,
        dropout=0.,
        maxlen=50
    ):
        """Initialize params."""
        super(StackedAttentionGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.batch_first = batch_first

        self.layers = []
        for i in range(num_layers):
            layer = GRUAttentionDot(
                input_size, rnn_size, batch_first=self.batch_first, maxlen=maxlen
            )
            self.add_module('layer_%d' % i, layer)
            self.layers += [layer]
            input_size = rnn_size

    def forward(self, input, hidden, ctx, ctx_mask=None, use_maxlen=False):
        """Propogate input through the layer."""
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            if ctx_mask is not None:
                ctx_mask = torch.ByteTensor(
                    ctx_mask.data.cpu().numpy().astype(np.int32).tolist()
                ).cuda()
            output, h_1_i = layer(input, h_0, ctx, ctx_mask, use_maxlen=use_maxlen)

            input = output

            if i != len(self.layers):
                input = self.dropout(input)

            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class DeepBidirectionalLSTM(nn.Module):
    r"""A Deep LSTM with the first layer being bidirectional."""

    def __init__(
        self, input_size, hidden_size,
        num_layers, dropout, batch_first
    ):
        """Initialize params."""
        super(DeepBidirectionalLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.batch_first = batch_first
        self.num_layers = num_layers

        self.bi_encoder = nn.LSTM(
            self.input_size,
            self.hidden_size // 2,
            1,
            bidirectional=True,
            batch_first=True,
            dropout=self.dropout
        )

        self.encoder = nn.LSTM(
            self.hidden_size,
            self.hidden_size,
            self.num_layers - 1,
            bidirectional=False,
            batch_first=True,
            dropout=self.dropout
        )

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder_bi = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))
        c0_encoder_bi = Variable(torch.zeros(
            2,
            batch_size,
            self.hidden_size // 2
        ))

        h0_encoder = Variable(torch.zeros(
            self.num_layers - 1,
            batch_size,
            self.hidden_size
        ))

        c0_encoder = Variable(torch.zeros(
            self.num_layers - 1,
            batch_size,
            self.hidden_size
        ))

        return (h0_encoder_bi.cuda(), c0_encoder_bi.cuda()), \
            (h0_encoder.cuda(), c0_encoder.cuda())

    def forward(self, input):
        """Propogate input forward through the network."""
        hidden_bi, hidden_deep = self.get_state(input)
        bilstm_output, (_, _) = self.bi_encoder(input, hidden_bi)
        return self.encoder(bilstm_output, hidden_deep)

class SoftDotAttention(nn.Module):
    """Soft Dot Attention.

    Ref: http://www.aclweb.org/anthology/D15-1166
    Adapted from PyTorch OPEN NMT.
    """

    def __init__(self, dim):
        """Initialize layer."""
        super(SoftDotAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def forward(self, input, context):
        """Propogate input through the network.

        input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        h_tilde = torch.cat((weighted_context, input), 1)

        h_tilde = self.tanh(self.linear_out(h_tilde))

        return h_tilde, attn


class GRUAttentionDot(nn.Module):
    r"""A long short-term memory (LSTM) cell with attention."""

    def __init__(self, input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3,maxlen=50):
        """Initialize params."""
        super(GRUAttentionDot, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.maxlen = maxlen
        self.dropout = dropout
        self.decoder_rnn = nn.GRU(input_size,
                                  hidden_size,
                                  num_layers,
                                  bidirectional=False,
                                  batch_first=True,
                                  dropout=self.dropout)
        self.attention_layer = SoftDotAttention(hidden_size)

    def forward(self, input, hidden, ctx, ctx_mask=None, use_maxlen=False):
        """Propogate input through the network."""
        def recurrence(input, hidden):
            """Recurrence helper."""
            hy , _ = self.decoder_rnn(input,hidden)
            h_tilde, alpha = self.attention_layer(hy.squeeze(1), ctx.transpose(0, 1))

            return h_tilde

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        if use_maxlen:
            steps = range(self.maxlen)
        else:
            steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i].unsqueeze(1), hidden.unsqueeze(0))
            output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


class Seq2SeqAttention(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        trg_vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        ctx_hidden_dim,
        attention_mode,
        batch_size,
        bidirectional=True,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.,
        gpu=False
    ):
        """Initialize model."""
        super(Seq2SeqAttention, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.ctx_hidden_dim = ctx_hidden_dim
        self.attention_mode = attention_mode
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            src_emb_dim
        )
        self.trg_embedding = nn.Embedding(
            trg_vocab_size,
            trg_emb_dim
        )

        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim
        self.encoder = nn.GRU(
            src_emb_dim,
            self.src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = StackedAttentionGRU(
            trg_emb_dim,
            trg_hidden_dim,
            nlayers,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(trg_hidden_dim, trg_vocab_size)
        self.gpu = gpu
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)
        if self.gpu:
            h0_encoder = h0_encoder.cuda()
        return h0_encoder

    def forward(self, input_src, input_trg, lengths=None, trg_mask=None, ctx_mask=None, gpu=False):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        trg_emb = self.trg_embedding(input_trg)

        self.h0_encoder = self.get_state(input_src)

        lengths = lengths.view(-1).data.tolist()
        
        packed_emb = pack_padded_sequence(src_emb, lengths,batch_first=True)
            
        src_h, src_h_t = self.encoder(packed_emb, self.h0_encoder)
        
        src_h = pad_packed_sequence(src_h)[0]
            
        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            
        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        ctx = src_h

        trg_h, _ = self.decoder(
            trg_emb,
            decoder_init_state,
            ctx,
            ctx_mask
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )
        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.trg_vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs


class Seq2SeqAttentionSharedEmbedding(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        emb_dim,
        vocab_size,
        src_hidden_dim,
        trg_hidden_dim,
        ctx_hidden_dim,
        attention_mode,
        batch_size,
        pad_token_src,
        pad_token_trg,
        bidirectional=True,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.,
    ):
        """Initialize model."""
        super(Seq2SeqAttentionSharedEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.ctx_hidden_dim = ctx_hidden_dim
        self.attention_mode = attention_mode
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1
        self.pad_token_src = pad_token_src
        self.pad_token_trg = pad_token_trg

        self.embedding = nn.Embedding(
            vocab_size,
            emb_dim,
            self.pad_token_src
        )

        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim

        self.encoder = nn.LSTM(
            emb_dim,
            self.src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder = LSTMAttentionDot(
            emb_dim,
            trg_hidden_dim,
            batch_first=True
        )

        self.encoder2decoder = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )
        self.decoder2vocab = nn.Linear(trg_hidden_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder.bias.data.fill_(0)
        self.decoder2vocab.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)

        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        c0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)

        return h0_encoder.cuda(), c0_encoder.cuda()

    def forward(self, input_src, input_trg, trg_mask=None, ctx_mask=None):
        """Propogate input through the network."""
        src_emb = self.embedding(input_src)
        trg_emb = self.embedding(input_trg)

        self.h0_encoder, self.c0_encoder = self.get_state(input_src)

        src_h, (src_h_t, src_c_t) = self.encoder(
            src_emb, (self.h0_encoder, self.c0_encoder)
        )

        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
            c_t = torch.cat((src_c_t[-1], src_c_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
            c_t = src_c_t[-1]

        decoder_init_state = nn.Tanh()(self.encoder2decoder(h_t))

        ctx = src_h.transpose(0, 1)

        trg_h, (_, _) = self.decoder(
            trg_emb,
            (decoder_init_state, c_t),
            ctx,
            ctx_mask
        )

        trg_h_reshape = trg_h.contiguous().view(
            trg_h.size()[0] * trg_h.size()[1],
            trg_h.size()[2]
        )

        decoder_logit = self.decoder2vocab(trg_h_reshape)
        decoder_logit = decoder_logit.view(
            trg_h.size()[0],
            trg_h.size()[1],
            decoder_logit.size()[1]
        )
        return decoder_logit

    def decode(self, logits):
        """Return probability distribution over words."""
        logits_reshape = logits.view(-1, self.vocab_size)
        word_probs = F.softmax(logits_reshape)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs

class Seq2SeqMono(nn.Module):
    """Container module with an encoder, deocder, embeddings."""

    def __init__(
        self,
        src_emb_dim,
        trg_emb_dim,
        src_vocab_size,
        trg_vocab_size_l1,
        trg_vocab_size_l2,
        src_hidden_dim,
        trg_hidden_dim,
        ctx_hidden_dim,
        attention_mode,
        batch_size,
        bidirectional=True,
        nlayers=2,
        nlayers_trg=2,
        dropout=0.,
        maxlen=50,
        gpu=False
    ):
        """Initialize model."""
        super(Seq2SeqMono, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size_l1 = trg_vocab_size_l1
        self.trg_vocab_size_l2 = trg_vocab_size_l2
        self.src_emb_dim = src_emb_dim
        self.trg_emb_dim = trg_emb_dim
        self.src_hidden_dim = src_hidden_dim
        self.trg_hidden_dim = trg_hidden_dim
        self.ctx_hidden_dim = ctx_hidden_dim
        self.attention_mode = attention_mode
        self.batch_size = batch_size
        self.bidirectional = bidirectional
        self.nlayers = nlayers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            src_emb_dim
        )
        self.trg_embedding_l1 = nn.Embedding(
            trg_vocab_size_l1,
            trg_emb_dim
        )
        self.trg_embedding_l2 = nn.Embedding(
            trg_vocab_size_l2,
            trg_emb_dim
        )

        self.src_hidden_dim = src_hidden_dim // 2 \
            if self.bidirectional else src_hidden_dim
        self.encoder = nn.GRU(
            src_emb_dim,
            self.src_hidden_dim,
            nlayers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=self.dropout
        )

        self.decoder_l1 = GRUAttentionDot(
            trg_emb_dim,
            trg_hidden_dim,
            nlayers,
            batch_first=True,
            maxlen=maxlen
        )
        
        self.decoder_l2 = GRUAttentionDot(
            trg_emb_dim,
            trg_hidden_dim,
            nlayers,
            batch_first=True,
            maxlen=maxlen
        )

        self.encoder2decoder1 = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )
        
        self.encoder2decoder2 = nn.Linear(
            self.src_hidden_dim * self.num_directions,
            trg_hidden_dim
        )
        
        self.decoder2vocab_l1 = nn.Linear(trg_hidden_dim, trg_vocab_size_l1)
        self.decoder2vocab_l2 = nn.Linear(trg_hidden_dim, trg_vocab_size_l2)
        self.gpu = gpu
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        self.src_embedding.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding_l1.weight.data.uniform_(-initrange, initrange)
        self.trg_embedding_l2.weight.data.uniform_(-initrange, initrange)
        self.encoder2decoder1.bias.data.fill_(0)
        self.encoder2decoder2.bias.data.fill_(0)
        self.decoder2vocab_l1.bias.data.fill_(0)
        self.decoder2vocab_l2.bias.data.fill_(0)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(0) \
            if self.encoder.batch_first else input.size(1)
        h0_encoder = Variable(torch.zeros(
            self.encoder.num_layers * self.num_directions,
            batch_size,
            self.src_hidden_dim
        ), requires_grad=False)
        if self.gpu:
            h0_encoder = h0_encoder.cuda()
        return h0_encoder

    def forward(self, input_src, input_trg, lengths=None, \
                trg_mask=None, ctx_mask=None, gpu=False, \
                l1_decoder=True, use_maxlen=False, unsup=False):
        """Propogate input through the network."""
        src_emb = self.src_embedding(input_src)
        if l1_decoder:
            if unsup:
                trg_emb = self.trg_embedding_l1(input_src)
            else:
                trg_emb = self.trg_embedding_l1(input_trg)
        else:
            if unsup:
                trg_emb = self.trg_embedding_l2(input_src)
            else:
                trg_emb = self.trg_embedding_l2(input_trg)

        self.h0_encoder = self.get_state(input_src)

        lengths = lengths.view(-1).data.tolist()
        
        packed_emb = pack_padded_sequence(src_emb, lengths,batch_first=True)
            
        src_h, src_h_t = self.encoder(packed_emb, self.h0_encoder)
        
        src_h = pad_packed_sequence(src_h)[0]
            
        if self.bidirectional:
            h_t = torch.cat((src_h_t[-1], src_h_t[-2]), 1)
        else:
            h_t = src_h_t[-1]
        
        if l1_decoder:
            decoder_init_state = nn.Tanh()(self.encoder2decoder1(h_t))
            ctx = src_h
            trg_h, _ = self.decoder_l1(
                trg_emb,
                decoder_init_state,
                ctx,
                ctx_mask,
                use_maxlen=use_maxlen
            )
    
            trg_h_reshape = trg_h.contiguous().view(
                trg_h.size()[0] * trg_h.size()[1],
                trg_h.size()[2]
            )
            decoder_logit = self.decoder2vocab_l1(trg_h_reshape)
            decoder_logit = decoder_logit.view(
                trg_h.size()[0],
                trg_h.size()[1],
                decoder_logit.size()[1]
            )
        else:
            decoder_init_state = nn.Tanh()(self.encoder2decoder2(h_t))
            ctx = src_h
            trg_h, _ = self.decoder_l2(
                trg_emb,
                decoder_init_state,
                ctx,
                ctx_mask,
                use_maxlen=use_maxlen
            )
    
            trg_h_reshape = trg_h.contiguous().view(
                trg_h.size()[0] * trg_h.size()[1],
                trg_h.size()[2]
            )
            decoder_logit = self.decoder2vocab_l2(trg_h_reshape)
            decoder_logit = decoder_logit.view(
                trg_h.size()[0],
                trg_h.size()[1],
                decoder_logit.size()[1]
            )
        return decoder_logit

    def decode(self, logits, l1_decoder=False):
        """Return probability distribution over words."""
        if l1_decoder:
            logits_reshape = logits.view(-1, self.trg_vocab_size_l1)
        else:
            logits_reshape = logits.view(-1, self.trg_vocab_size_l2)
        word_probs = F.softmax(logits)
        word_probs = word_probs.view(
            logits.size()[0], logits.size()[1], logits.size()[2]
        )
        return word_probs

