import csv

import numpy as np
from neural_network.cloze_stories.data import ClozeStoriesData
from neural_network.cloze_stories.data.models import Dataset, Story, DataSplit
import logging

logger = logging.getLogger('reader')


class ClozeStoriesCSVData(ClozeStoriesData):
    def __init__(self, config, config_global, logger):

        """This is a reader that can read the offical ROC stories csv file for training
        and the Story Cloze Test stories csv files for evaluation.

        The reader is applied on the following data:

        1. Augmented ROCStories -- that is, generated either with the Random Shuffle or KDE Sampling method.
        2. Story Cloze Test validation set - development set.
        3. Story Cloze Test test set - for computing statistics

        This reader is used for the BILSTM-V model

        """

        super(ClozeStoriesCSVData, self).__init__(config, config_global, logger)

        self.train_stories_csv_path = self.config['train_stories_csv']
        self.valid_csv_path = self.config['valid_csv']
        self.test_csv_path = self.config['test_csv']

    def load_dataset(self):

        """
        Prepare the training and validation data for training.
        Note that we split both the development and training data. The training data is split in case we want to use it
        to augment the Story Cloze Test validation data.
        Training data is assumed to be generated using either Random Shuffle or KDE sampling method.

        """

        train_split_size = self.config["train_split_size"]
        training_stories = self.load_stories_sct_fmt(self.train_stories_csv_path)
        training_stories_large, training_stories_small = self.split_stories(training_stories, train_split_size)

        val_split_size = self.config["val_split_size"]
        valid_stories = self.load_stories_sct_fmt(self.valid_csv_path)
        valid_stories_large, valid_stories_small = self.split_stories(valid_stories, val_split_size)

        test_stories = self.load_stories_sct_fmt(self.test_csv_path)

        num_val_stories_for_training = len(valid_stories_large)
        num_val_stories_for_validation = len(valid_stories_small)
        num_train_stories_for_training = len(training_stories_large)
        num_test_stories = len(test_stories)

        # augmenting the larger part of the validation stories with a portion of training set instances
        valid_stories_large.extend(training_stories_large)
        train_stories_permuted_indices = np.random.permutation(len(valid_stories_large))

        # augmented training data
        training_data = DataSplit([valid_stories_large[i] for i in train_stories_permuted_indices])
        valid_data = DataSplit(valid_stories_small)
        test_data = DataSplit(test_stories)

        logger.debug("# train stories: %d (trainset), %d (valset); # val stories %d (valset)" %
                    (num_train_stories_for_training, num_val_stories_for_training, num_val_stories_for_validation)
                    )
        logger.debug("# test stories: %d" % num_test_stories)

        self.dataset = Dataset(training_data, valid_data, test_data)

    def split_stories(self, stories, split_ratio):

        """
        Split the stories (list) accoring to the split ratio.
        :param stories: a list of stories
        :param split_ratio: split ratio (float)
        :return: two story lists of sizes: split_ratio * len(stories), (1.0 - split_ratio) * len(stories)
        """

        num_train_instances = len(stories)
        indices = np.random.permutation(num_train_instances)
        split_size = int(split_ratio * num_train_instances)

        if split_size == 0:
            return [], [stories[i] for i in indices]

        training_idx, val_idx = indices[:split_size], indices[split_size:]
        training_data = [stories[i] for i in training_idx]
        valid_data = [stories[i] for i in val_idx]
        return training_data, valid_data

    def load_stories_sct_fmt(self, path):

        """

        Load data in the Story Cloze Test format:

        story_id    snt1    snt2    snt3    snt4    ending1     ending2     label   feature1    feature2    ...

        Note that label is either "1" or "2", so we subtract "1" to make it binary
        :rtype: stories (list)

        """

        stories = []

        with open(path, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            header = next(reader)
            num_elems = len(header)
            for row in list(reader):
                story_id = row[0]
                sentences = []
                for text in row[1:5]:
                    sentences.append(self.get_sentence(text))
                potential_endings = []
                for text in row[5:7]:
                    potential_endings.append(self.get_sentence(text))

                correct_endings = [int(row[7]) - 1]
                feature_values1 = []
                feature_values2 = []
                if len(row) >= 9:
                    for idx in range(8, num_elems):
                        value = row[idx]
                        if "E1" in header[idx]:
                            feature_values1.append(float(value))
                        elif "E2" in header[idx]:
                            feature_values2.append(float(value))
                        else:
                            feature_values1.append(float(value))
                            feature_values2.append(float(value))

                potential_endings[0].metadata['feature_values'] = feature_values1
                potential_endings[1].metadata['feature_values'] = feature_values2
                stories.append(Story(sentences, potential_endings, correct_endings, id = story_id))
        return stories

    @property
    def n_features(self):
        return len(self.dataset.train.stories[0].potential_endings[0].metadata.get('feature_values', []))


component = ClozeStoriesCSVData
