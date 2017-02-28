from neural_network.util import unique_items


class MetadataObject(object):
    def __init__(self):
        self.metadata = dict()


class Dataset(MetadataObject):
    def __init__(self, train, valid, test):
        """

        :param train: DataSplit
        :param valid: DataSplit
        :param test: DataSplit
        """
        super(Dataset, self).__init__()
        self.train = train
        self.valid = valid
        self.test = test

        self._vocab = None

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = unique_items(self.train.vocab + self.valid.vocab + self.test.vocab)
        return self._vocab


class DataSplit(MetadataObject):
    def __init__(self, stories):
        """

        :type stories: list[Story]
        """
        super(DataSplit, self).__init__()
        self.stories = stories

    @property
    def vocab(self):
        return unique_items([i for story in self.stories for i in story.vocab])


class Story(MetadataObject):
    def __init__(self, sentences, potential_endings=None, correct_endings=None, id = None):
        """

        :type sentences: list[Sentence]
        :type potential_endings: list[Sentence]
        :type correct_endings: list[int]
        """
        super(Story, self).__init__()
        self.sentences = sentences
        self.potential_endings = potential_endings or []
        self.correct_endings = correct_endings or []
        self.id = id

    @property
    def vocab(self):
        return unique_items([i for sentence in self.sentences + self.potential_endings for i in sentence.vocab])


class Sentence(MetadataObject):
    def __init__(self, text, tokens):
        """

        :type text: basestring
        :type tokens: list[basestring]
        """
        super(Sentence, self).__init__()
        self.text = text
        self.tokens = tokens

    @property
    def vocab(self):
        return unique_items(self.tokens)
