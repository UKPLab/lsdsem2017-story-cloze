# coding=utf-8

import math

import numpy as np

from neural_network.cloze_stories.train import ClozeTraining


class ClozeTrainingRandomShuffle(ClozeTraining):
    """This is a training method that creates negative examples by randomly shuffling the stories

    """

    def __init__(self, config, config_global, logger):
        super(ClozeTrainingRandomShuffle, self).__init__(config, config_global, logger)
        self.batch_i = 0
        self.examples_positive = []
        self.epoch_random_indices = None

    def prepare_next_epoch(self, model, data, sess, epoch):
        super(ClozeTrainingRandomShuffle, self).prepare_next_epoch(model, data, sess, epoch)
        self.batch_i = 0

        # we prepare all positive examples only once
        if len(self.examples_positive) == 0:
            self.logger.debug('Preparing training examples (positive only)')
            self.examples_positive = []
            for story in data.dataset.train.stories:
                beginning_good_vec = data.get_item_vector(story.sentences[:-1], self.sentence_length * 4)
                end_good_vec = data.get_item_vector([story.sentences[-1]], self.sentence_length)
                self.examples_positive.append(
                    (beginning_good_vec, end_good_vec, story)
                )

        # shuffle the indices of each batch
        self.epoch_random_indices = np.random.permutation(len(self.examples_positive))

    def get_n_batches(self):
        return int(math.ceil(len(self.examples_positive) / self.batchsize))

    def get_next_batch(self, model, data, sess):
        """For each positive example, we create a negative example by
        randomly shuffling the sentences of the story

        :return: story beginning, story end, label
        :rtype: list, list, list
        """
        indices = self.epoch_random_indices[self.batch_i * self.batchsize / 2: (self.batch_i + 1) * self.batchsize / 2]
        incomplete_data_epoch = [self.examples_positive[i] for i in indices]

        batch_story_begin = []
        batch_story_end = []
        batch_label = []

        for good_story_begin_vecs, good_story_end_vecs, story in incomplete_data_epoch:
            batch_story_begin.append(good_story_begin_vecs)
            batch_story_end.append(good_story_end_vecs)
            batch_label.append([1, 0])

            # To create negative examples, we first flip the last sentence with a sentence from the beginning
            shuffled_story = story.sentences
            flip = np.random.randint(0, 4)
            shuffled_story[4], shuffled_story[flip] = shuffled_story[flip], shuffled_story[4]
            # then we shuffle the beginning
            shuffled_story = list(np.random.permutation(shuffled_story[:4])) + shuffled_story[4:]

            shuffled_story_begin_vecs = data.get_item_vector(shuffled_story[:-1], self.sentence_length * 4)
            shuffled_story_end_vecs = data.get_item_vector([shuffled_story[-1]], self.sentence_length)
            batch_story_begin.append(shuffled_story_begin_vecs)
            batch_story_end.append(shuffled_story_end_vecs)
            batch_label.append([0, 1])

        batch_shuffle = np.random.permutation(len(batch_label))
        batch_story_begin = [batch_story_begin[i] for i in batch_shuffle]
        batch_story_end = [batch_story_end[i] for i in batch_shuffle]
        batch_label = [batch_label[i] for i in batch_shuffle]

        self.batch_i += 1
        return batch_story_begin, batch_story_end, np.array([[]] * len(batch_story_begin)), batch_label


component = ClozeTrainingRandomShuffle
