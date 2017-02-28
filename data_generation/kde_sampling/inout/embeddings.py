import linecache
import logging
import numpy as np
import progressbar as pb


logger = logging.getLogger(__name__)


class EmbeddingsFromDisk:
    def __init__(self, pretrained_embeddings_path, uncased=False):
        self.embeddings_path = pretrained_embeddings_path
        self.index = self._create_index(pretrained_embeddings_path)
        self.uncased = uncased
        assert hasattr(self, 'embedding_dimension')

    def _create_index(self, pretrained_embeddings_path):
        """
        Returns a dict which maps from words to line numbers in the pretrained embeddings file.
        Determines the embedding dimension along the way.
        :param pretrained_embeddings_path:
        :return:
        """
        word_embedding_line_index = {}
        with open(pretrained_embeddings_path, 'r') as embeddings:
            line_number = 1
            for line in embeddings:
                rows = line.split(" ")
                word = rows[0]
                word_embedding_line_index[word] = line_number

                if line_number == 1:
                    self.embedding_dimension = len(rows[1:])

                line_number += 1
        return word_embedding_line_index

    def case_adjusted(self, word):
        return word.lower() if self.uncased else word

    def has_embedding(self, word):
        return self.case_adjusted(word) in self.index

    def get_embedding(self, word):
        """
        Given a word, returns an embedding as a numpy array of floats.
        :param word:
        :return:
        """
        if not self.has_embedding(word):
            raise ValueError("No embedding known for " + word)

        line_no = self.index[self.case_adjusted(word)]
        line = linecache.getline(self.embeddings_path, line_no).strip()
        vec = [float(s) for s in line.split(" ")[1:]]
        return np.array(vec, dtype=np.dtype('f8'))

    def get_random_embedding(self):
        """
        Returns a random embedding (with values in [-1, 1]).
        :return:
        """
        return 2 * np.random.sample(self.embedding_dimension) - 1


class EmbeddingsFromMemory(EmbeddingsFromDisk):
    def __init__(self, pretrained_embeddings_path, uncased=False):
        super().__init__(pretrained_embeddings_path, uncased)
        self.embeddings = self._load_pretrained_embeddings(pretrained_embeddings_path)

        # pick an arbitrary embedding to determine the embedding dimension
        self.embedding_dimension = len(next(iter(self.embeddings.values())))

    def _load_pretrained_embeddings(self, path):
        embedding_dict = {}
        with open(path, 'r') as embeddings_file:
            # determine the embedding dimension: here we daringly assume that the first entry in the embeddings file
            # is only one word (think "aardvark", not a special case like "or name@domain")
            first_line = embeddings_file.readline()
            self.embedding_dimension = len(first_line.split()) - 1

            # determine the number of lines in the file (+1 since we already read line no. 1)
            number_of_lines = 1 + sum(1 for _ in embeddings_file)

            # now start actually reading the contents from the first line
            embeddings_file.seek(0)
            bar = pb.ProgressBar(max_value=number_of_lines)
            for line in bar(embeddings_file):
                rows = line.split()

                # embedding values lie within the last n rows for n = self.embedding_dimension
                embedding = np.array([float(s) for s in rows[-self.embedding_dimension:]], dtype=np.dtype('f8'))

                # word(s) are the remaining rows in front, join them by spaces
                word = " ".join(rows[:-self.embedding_dimension])
                embedding_dict[word] = embedding

        return embedding_dict

    def has_embedding(self, word):
        return self.case_adjusted(word) in self.embeddings

    def get_embedding(self, word):
        if not self.has_embedding(word):
            raise ValueError("No embedding known for " + word)
        return self.embeddings[self.case_adjusted(word)]
