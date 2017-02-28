from __future__ import division

import numpy as np
import progressbar
import tensorflow as tf

import neural_network
import os

class ClozeTraining(neural_network.Training):
    def __init__(self, config, config_global, logger):
        super(ClozeTraining, self).__init__(config, config_global, logger)

        self.sentence_length = self.config_global['sentence_length']
        self.n_epochs = self.config['epochs']
        self.batchsize = self.config['batchsize']
        self.batchsize_valid = self.config.get('batchsize_valid', self.batchsize)
        self.dropout_keep_prob = 1.0 - self.config.get('dropout', 0.0)
        self.early_stopping_patience = self.config.get('early_stopping_patience', self.n_epochs)

        self.initial_learning_rate = self.config.get('initial_learning_rate', 1.1)
        self.dynamic_learning_rate = self.config.get('dynamic_learning_rate', True)
        self.epoch_learning_rate = self.initial_learning_rate

        # tensorboard summary writer
        self.global_step = 0
        self.train_writer = None

        # other properties
        self.valid_data = None

        self.checkpoint_dir = self.config_global.get('checkpoint_dir',
                                              os.path.abspath(os.path.join(os.environ['HOME'], "sc_dl_models")))

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)


    def start(self, model, data, sess):
        if 'tensorflow_log_dir' in self.config_global:
            self.train_writer = tf.train.SummaryWriter(self.config_global['tensorflow_log_dir'], sess.graph)

        learning_rate = tf.placeholder(tf.float32, shape=[])
        optimizer_name = self.config.get('optimizer', 'sgd')
        if optimizer_name == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        elif optimizer_name == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate)
        elif optimizer_name == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(learning_rate)
        else:
            raise Exception('No such optimizer: {}'.format(optimizer_name))

        train = optimizer.minimize(model.loss)
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())
        best_val_score = 0.0
        i = 0
        for epoch in range(1, self.n_epochs + 1):
            self.logger.info('Epoch {}/{}'.format(epoch, self.n_epochs))

            self.logger.debug('Preparing epoch')
            self.prepare_next_epoch(model, data, sess, epoch)

            bar = _create_progress_bar('loss')
            train_losses = []  # used to calculate the epoch train loss
            recent_train_losses = []  # used to calculate the display loss

            self.logger.debug('Training')
            for _ in bar(range(self.get_n_batches())):
                self.global_step += self.batchsize
                train_story_begin, train_story_end, train_features, train_label = self.get_next_batch(model, data, sess)

                _, loss, loss_individual, summary = sess.run(
                    [train, model.loss, model.loss_individual, model.summary],
                    feed_dict={
                        learning_rate: self.epoch_learning_rate,
                        model.input_story_begin: train_story_begin,
                        model.input_story_end: train_story_end,
                        model.input_label: train_label,
                        model.input_features: train_features,
                        model.dropout_keep_prob: self.dropout_keep_prob
                    })
                recent_train_losses = ([loss] + recent_train_losses)[:20]
                train_losses.append(loss)
                bar.dynamic_messages['loss'] = np.mean(recent_train_losses)
                self.add_summary(summary)
            self.logger.info('train loss={:.6f}'.format(np.mean(train_losses)))

            self.logger.info('Now calculating validation score')
            valid_score = self.calculate_validation_score(sess, model, data)
            if valid_score > best_val_score:
                score_val = np.around(valid_score, decimals=3)
                model_name = "%s/%0.3f.model.ckpt" % (self.checkpoint_dir, score_val)
                self.logger.info('Saving the model into %s' % model_name)
                saver.save(sess, model_name)
                best_val_score = valid_score
            self.logger.info('Score={:.4f}'.format(valid_score))

    def add_summary(self, summary):
        if self.train_writer:
            self.train_writer.add_summary(summary, self.global_step)

    def calculate_validation_score(self, sess, model, data):
        if self.valid_data is None:
            self.logger.info('Creating validation data')
            self.valid_data = []  # a list of beginning-vecs, ending-1-vecs, ending-2-vecs, label
            for story in data.dataset.valid.stories:
                beginning_vecs = data.get_item_vector(story.sentences, self.sentence_length * 4)
                ending_1_vecs = data.get_item_vector([story.potential_endings[0]], self.sentence_length)
                ending_1_features = np.array(story.potential_endings[0].metadata.get('feature_values', []))
                ending_2_vecs = data.get_item_vector([story.potential_endings[1]], self.sentence_length)
                ending_2_features = np.array(story.potential_endings[1].metadata.get('feature_values', []))
                label = story.correct_endings[0]
                self.valid_data.append(
                    (beginning_vecs, ending_1_vecs, ending_1_features, ending_2_vecs, ending_2_features, label)
                )
            self.logger.info('Done')

        bar = _create_progress_bar('score')

        correct = 0
        for i, (beginning_vecs, ending_1_vecs, ending_1_features, ending_2_vecs, ending_2_features, label) \
                in enumerate(bar(self.valid_data), start=1):
            # TODO this is not very effective. Use larger batches.
            predict, = sess.run([model.predict], feed_dict={
                model.input_story_begin: [beginning_vecs] * 2,
                model.input_story_end: [ending_1_vecs, ending_2_vecs],
                model.input_features: [ending_1_features, ending_2_features],
                model.dropout_keep_prob: 1.0
            })
            label_prediction = 0 if predict[0][0] > predict[1][0] else 1
            if label_prediction == label:
                correct += 1
            bar.dynamic_messages['score'] = correct / float(i)

        return correct / float(len(self.valid_data))

    def prepare_next_epoch(self, model, data, sess, epoch):
        """Prepares the next epoch, especially the batches"""
        self.epoch_learning_rate = self.initial_learning_rate
        if self.dynamic_learning_rate:
            self.epoch_learning_rate /= float(epoch)

    def get_n_batches(self):
        """:return: the number of batches"""
        raise NotImplementedError()

    def get_next_batch(self, model, data, sess):
        """Return the training data for the next batch

        :return: questions, good answers, bad answers
        :rtype: list, list, list
        """
        raise NotImplementedError()


def _create_progress_bar(dynamic_msg=None):
    widgets = [
        ' [batch ', progressbar.SimpleProgress(), '] ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') '
    ]
    if dynamic_msg is not None:
        widgets.append(progressbar.DynamicMessage(dynamic_msg))
    return progressbar.ProgressBar(widgets=widgets)
