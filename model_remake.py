import onmt
import torch
import torch.nn as nn

from onmt.Models import EncoderBase, InputFeedRNNDecoder
from onmt.modules import Embeddings
from torch.nn import Embedding
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class FixedEmbedding(Embedding):
    """Fixed embeddings.

    See torch.nn.Embedding() for details. The main differences are:
        - Input assumes additional 'nfeats' dimension for OpenNMT
            compatibility.
        - Embeddings are fixed during training.
    """
    def __init__(self, *args, **kwargs):
        super(FixedEmbedding, self).__init__(*args, **kwargs)
        self.weight.requires_grad = False

    def load(self, embedding_tensor):
        """Loads pretrained embeddings from a tensor.

        Args:
            embedding_tensor: (LongTensor) vocab_size x embedding_size.
        """
        self.weight.data.copy_(embedding_tensor)

    def forward(self, input):
        """
        Args:
            input: (LongTensor) len x batch_size x nfeats. The last dimension
            is included for compatibility with OpenNMT, and should always be 1.
        """
        length, batch_size, nfeats = input.size()
        assert nfeats == 1, "use onmt.modules.Embeddings instead"
        input = torch.squeeze(input, 2)
        out = super(FixedEmbedding, self).forward(input)
        return out

    @property
    def embedding_size(self):
        """Included for OpenNMT compatibility."""
        return self.embedding_dim


class RNNEncoder(EncoderBase):
    """Bilingual RNN encoder.

    Params:
        rnn_type: (string) Type of RNN to use. Any class in torch.nn.rnn.
        bidirectional: (bool) Whether to use bidirectional rnn.
        num_layers: (int) Number of layers.
        hidden_size: (int) Number of hidden units.
        dropout: (float) Dropout rate.
        l1_embeddings: (Embeddings) Retrieves embeddings for words in language
            l1.
        l2_embeddings: (Embeddings) Retrieves embeddings for words in language
            l2.
    """
    def __init__(self, rnn_type, bidirectional, num_layers, hidden_size,
                 dropout, l1_embeddings, l2_embeddings):
        super(RNNEncoder, self).__init__()

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        assert l1_embeddings.embedding_size == l2_embeddings.embedding_size
        hidden_size = hidden_size // num_directions
        self.l1_embeddings = l1_embeddings
        self.l2_embeddings = l2_embeddings
        self.no_pack_padded_seq = False

        self.rnn = getattr(nn, rnn_type)(
                input_size=l1_embeddings.embedding_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional)

    def forward(self, input, lang, lengths=None, hidden=None):
        """
        Args:
            input: (LongTensor) len x batch_size x nfeats.
            lang: (string) Either 'l1' or 'l2'.
            lengths: (LongTensor) batch_size. Lengths of each sentence in
                batch.
            hidden: Initial hidden state.
        """
        if lang=='l1':
            emb = self.l1_embeddings(input)
        elif lang=='l2':
            emb = self.l2_embeddings(input)
        else:
            raise ValueError('lang')
        s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Variable.
            lengths = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths)

        outputs, hidden_t = self.rnn(packed_emb, hidden)

        if lengths is not None and not self.no_pack_padded_seq:
            outputs = unpack(outputs)[0]

        return hidden_t, outputs


class Model(nn.Module):
    """The unsupervised neural machine translation model.

    Params:
        l1_vocab_size: (int) Number of words in language l1's vocabulary.
        l2_vocab_size: (int) Number of words in language l2's vocabulary.
        embedding_size: (int) Size of embeddings.
        hidden_size: (int) Size of hidden representation.
        rnn_type: (string) Type of RNN to use. Any class in torch.nn.rnn.
        n_layers_encoder: (int) Number of hidden layers.
        n_layers_decoder: (int) Number of hidden layers.
        dropout: (float) Dropout rate.
    """
    def __init__(self, l1_vocab_size, l2_vocab_size, embedding_size,
                 hidden_size, rnn_type, n_layers_encoder, n_layers_decoder,
                 dropout):
        super(Model, self).__init__()
        self.l1_vocab_size = l1_vocab_size
        self.l2_vocab_size = l2_vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        self.n_layers_encoder = n_layers_encoder
        self.n_layers_decoder = n_layers_decoder
        self.dropout = dropout

        # Embed
        self.l1_embeddings = FixedEmbedding(l1_vocab_size, embedding_size)
        self.l2_embeddings = FixedEmbedding(l2_vocab_size, embedding_size)
        # Encode
        self.encoder = RNNEncoder(
            rnn_type=rnn_type,
            bidirectional=True,
            num_layers=n_layers_encoder,
            hidden_size=hidden_size,
            dropout=dropout,
            l1_embeddings=self.l1_embeddings,
            l2_embeddings=self.l2_embeddings)
        # Decode
        self.l1_decoder = InputFeedRNNDecoder(
            rnn_type=rnn_type,
            bidirectional_encoder=True,
            num_layers=n_layers_decoder,
            hidden_size=hidden_size,
            attn_type='general',
            coverage_attn=None, # Not supported.
            context_gate=None,
            copy_attn=None, # Not supported.
            dropout=dropout,
            embeddings=self.l1_embeddings)
        self.l2_decoder = InputFeedRNNDecoder(
            rnn_type=rnn_type,
            bidirectional_encoder=True,
            num_layers=n_layers_decoder,
            hidden_size=hidden_size,
            attn_type='general',
            coverage_attn=None, # Not supported.
            context_gate=None,
            copy_attn=None, # Not supported.
            dropout=dropout,
            embeddings=self.l2_embeddings)
        # Project
        self.l1_to_vocab = nn.Linear(hidden_size, l1_vocab_size)
        self.l2_to_vocab = nn.Linear(hidden_size, l2_vocab_size)

    def forward(self, src, src_lang, lengths, tgt, tgt_lang, dec_state=None):
        """
        Args:
            src: (LongTensor) src_len x batch_size x nfeats. Sentences in source
                language.
            src_lang: (String) Language of src. Either 'l1' or 'l2'.
            lengths: (LongTensor) batch_size. Lengths of sentences in src.
            tgt: (LongTensor) tgt_len x batch_size x nfeats. Sentences in
                target language.
            tgt_lang: (String) Language of tgt. Either 'l1' or 'l2'.
            dec_state: (onmt.Models.DecoderState) Optional. See onmt.Models for
                further details.

        Returns:
            logits: (FloatTensor) Output logits.
            attn: (FloatTensor) Attentions over encoder outputs.
            dec_state: (onmt.Models.DecoderState) To be documented.
        """
        if tgt_lang == 'l1':
            decoder = self.l1_decoder
            fc = self.l1_to_vocab
        elif tgt_lang == 'l2':
            decoder = self.l2_decoder
            fc = self.l2_to_vocab
        else:
            raise ValueError('tgt_lang')

        tgt = tgt[:-1]
        enc_hidden, context = self.encoder(src, src_lang, lengths)
        enc_state = decoder.init_decoder_state(src, context, enc_hidden)
        out, dec_state, attns = decoder(tgt, context,
                                        enc_state if dec_state is None
                                        else dec_state)
        logits = fc(out)
        return logits, attns, dec_state


if __name__ == '__main__':
    # Create model.
    mdl = Model(l1_vocab_size=10, l2_vocab_size=10, embedding_size=2,
                hidden_size=2, rnn_type="GRU", n_layers_encoder=2,
                n_layers_decoder=2, dropout=0.3)
    # Example input data.
    src_sent = torch.autograd.Variable(torch.LongTensor([[1, 1, 1],
                                                         [2, 0, 1],
                                                         [5, 1, 1],
                                                         [1, 1, 0],
                                                         [1, 0, 0]]))
    print(src_sent.size())
    lengths = torch.LongTensor([5, 4, 3])
    # Verify that forward computation works.
    out, _, _ = mdl(
        src=src_sent,
        src_lang='l1',
        lengths=lengths,
        tgt=src_sent,
        tgt_lang='l2')
    print(out)

