import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer

import neural_network


class ClozeModel(neural_network.Model):
    def __init__(self, config, config_global, logger):
        super(ClozeModel, self).__init__(config, config_global, logger)
        self.__summary = None

        self.trainable_embeddings = self.config.get('trainable_embeddings', True)
        self.sentence_length = self.config_global['sentence_length']
        self.embedding_size = self.config_global['embedding_size']

    def build_input(self, data, sess):
        self.input_story_begin = tf.placeholder(tf.int32, [None, self.sentence_length * 4])
        self.input_story_end = tf.placeholder(tf.int32, [None, self.sentence_length])
        self.input_features = tf.placeholder(tf.float32, [None, data.n_features])
        self.input_label = tf.placeholder(tf.int32, [None, 2])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embeddings_init = tf.constant_initializer(data.embeddings)
            embeddings_weight = tf.get_variable("embeddings", data.embeddings.shape, dtype=tf.float32,
                                                initializer=embeddings_init,
                                                trainable=self.trainable_embeddings)

            self.embeddings_story_begin = tf.nn.embedding_lookup(embeddings_weight, self.input_story_begin)
            self.embeddings_story_end = tf.nn.embedding_lookup(embeddings_weight, self.input_story_end)

    def create_outputs(self, predict):
        self.loss_individual = tf.nn.softmax_cross_entropy_with_logits(predict, self.input_label)
        self.loss = tf.reduce_mean(self.loss_individual)

        self.predict = tf.nn.softmax(predict)
        tf.scalar_summary('Loss', self.loss)

    @property
    def summary(self):
        if self.__summary is None:
            self.__summary = tf.merge_all_summaries(key='summaries')
        return self.__summary


def weight_variable(name, shape):
    return tf.get_variable(name, shape=shape, initializer=xavier_initializer())


def bias_variable(name, shape, value=0.1):
    return tf.get_variable(name, shape, initializer=tf.constant_initializer(value))


def non_zero_tokens(tokens):
    """Receives a vector of tokens (float) which are zero-padded. Returns a vector of the same size, which has the value
    1.0 in positions with actual tokens and 0.0 in positions with zero-padding.

    :param tokens:
    :return:
    """
    return tf.ceil(tokens / tf.reduce_max(tokens, [1], keep_dims=True))
