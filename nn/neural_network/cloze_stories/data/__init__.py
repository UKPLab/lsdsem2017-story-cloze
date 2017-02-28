import re

import numpy as np

import neural_network
from neural_network.cloze_stories.data.models import Sentence
from neural_network.util import read_embeddings


class ClozeStoriesData(neural_network.Data):
    def __init__(self, config, config_global, logger):
        super(ClozeStoriesData,self).__init__(config, config_global, logger)

        self.lowercased = self.config.get('lowercased', False)
        self.embedding_size = self.config_global['embedding_size']
        self.embeddings_path = self.config.get('embeddings_path')
        self.map_oov = self.config.get('map_oov', False)

        self.vocab_to_index = dict()  # a dict that matches each token to an integer position for the embeddings
        self.embeddings = None  # numpy array that contains all relevant embeddings

        self.dataset = None
        # :type Dataset:

    def setup(self):
        self.load_dataset()
        self.setup_embeddings()

    def load_dataset(self):
        raise NotImplementedError()

    def setup_embeddings(self):
        if 'embeddings_path' in self.config:
            # load the initial embeddings
            self.logger.info('Fetching the dataset vocab')
            vocab = self.dataset.vocab
            self.logger.info('Loading embeddings (vocab size={})'.format(len(vocab)))
            embeddings_dict = read_embeddings(self.embeddings_path, vocab, self.logger)

            zero_padding = np.zeros((self.embedding_size,))
            oov = np.random.uniform(-1.0, 1.0, [self.embedding_size, ])
            embeddings = [zero_padding, oov]

            n_oov = 0
            for token in self.dataset.vocab:
                embedding_dict_item = embeddings_dict.get(token, None)
                if embedding_dict_item is not None:
                    self.vocab_to_index[token] = len(embeddings)
                    embeddings.append(embedding_dict_item)
                else:
                    n_oov += 1
                    if self.map_oov:
                        self.vocab_to_index[token] = 1  # oov
                    else:
                        # for each oov, we create a new random vector
                        self.vocab_to_index[token] = len(embeddings)
                        embeddings.append(np.random.uniform(-1.0, 1.0, [self.embedding_size, ]))

            self.embeddings = np.array(embeddings)
            self.logger.debug('OOV tokens: {}'.format(n_oov))
        else:
            self.vocab_to_index = dict([(t, i) for (i, t) in enumerate(self.dataset.vocab, start=1)])
            self.embeddings = np.append(
                np.zeros((1, self.embedding_size)),  # zero-padding
                np.random.uniform(-1.0, 1.0, [len(self.dataset.vocab), self.embedding_size]),
                axis=0
            )

    def get_sentence(self, text):
        """

        :param text: basestring
        :rtype: Sentence
        """
        text = re.sub('[^0-9a-zA-Z ]+', '', text)
        if self.lowercased:
            text = text.lower()

        tokens = text.split()
        return Sentence(text, tokens)

    def get_item_vector(self, sentences, max_len):
        """Calculates the actual vector for the text items that can be fed into tensorflow

        :param sentences:
        :param max_len:
        :return:
        """
        tokens = []
        for sentence in sentences:
            if len(tokens) < max_len:
                tokens += [self.vocab_to_index[t] for t in sentence.tokens]
        tokens = tokens[:max_len]

        # zero-pad to max_len
        tokens_padded = tokens + [0 for _ in range(max_len - len(tokens))]
        return tokens_padded

    @property
    def n_features(self):
        return 0
