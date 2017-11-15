"""Embedding lookup object"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class EmbeddingLookup(object):

    def __init__(self, tokens, embeddings):
        """Initializes EmbeddingLookup object.

        Args:
            tokens: List of tokens.
            embeddings: List of corresponding token embeddings.
        """
        self._tokens = tokens
        self._index_lookup = {token: i for i, token in enumerate(tokens)}
        self._embeddings = np.array(embeddings)

    @classmethod
    def from_file(cls, filename, skip=1):
        """Loads embeddings from a file.
        
        Args:
            filename: Name of file to load embeddings from.
            skip: Number of lines to skip.

        Returns:
            An EmbeddingLookup object.
        """
        with open(filename, 'r') as f:
            tokens = []
            vectors = []
            for i, line in enumerate(f):
                if i < skip:
                    continue
                token, *vector = line.split()
                vector = [float(x) for x in vector]
                tokens.append(token)
                vectors.append(vector)
        embedding_lookup = cls(tokens, vectors)
        return embedding_lookup

    def embed(self, tokens):
        """Embeds a list of tokens.

        Args:
            tokens: A list of tokens.

        Returns:
            A numpy array containing the token embeddings.
        """
        indices = np.array([self._index_lookup[token] for token in tokens])
        return self._embeddings[indices, :]

