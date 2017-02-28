# coding=utf-8

import math

import numpy as np

from neural_network.cloze_stories.train import ClozeTraining


class ClozeTrainingFeatures(ClozeTraining):
    """This is a training method that uses

    """

    def __init__(self, config, config_global, logger):
        super(ClozeTrainingFeatures, self).__init__(config, config_global, logger)
        self.batch_i = 0
        self.examples = []
        self.epoch_random_indices = None

    def prepare_next_epoch(self, model, data, sess, epoch):
        super(ClozeTrainingFeatures, self).prepare_next_epoch(model, data, sess, epoch)
        self.batch_i = 0

        # we prepare all examples only once
        if len(self.examples) == 0:
            self.logger.debug('Preparing training examples')
            self.examples = []
            for story in data.dataset.train.stories:
                beginning_vec = data.get_item_vector(story.sentences, self.sentence_length * 4)
                for potential_ending_i, potential_ending in enumerate(story.potential_endings):
                    label = [1, 0] if potential_ending_i in story.correct_endings else [0, 1]
                    ending_vec = data.get_item_vector([potential_ending], self.sentence_length)
                    feature_values = np.array(potential_ending.metadata['feature_values'])
                    self.examples.append((beginning_vec, ending_vec, feature_values, label))

        # shuffle the indices of each batch
        self.epoch_random_indices = np.random.permutation(len(self.examples))

    def get_n_batches(self):
        return int(math.ceil(len(self.examples) / self.batchsize))

    def get_next_batch(self, model, data, sess):
        """We just return the next batch data here

        :return: story beginning, story end, label
        :rtype: list, list, list
        """
        indices = self.epoch_random_indices[self.batch_i * self.batchsize: (self.batch_i + 1) * self.batchsize]
        data = [self.examples[i] for i in indices]
        batch_story_begin, batch_story_end, batch_feature_values, batch_label = zip(*data)
        self.batch_i += 1
        return batch_story_begin, batch_story_end, batch_feature_values, batch_label


component = ClozeTrainingFeatures
