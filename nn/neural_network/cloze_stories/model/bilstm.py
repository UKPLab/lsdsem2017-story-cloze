import tensorflow as tf
from tensorflow.python.ops import rnn_cell

from neural_network.cloze_stories.model import ClozeModel, weight_variable, bias_variable, non_zero_tokens


class BiLSTMModel(ClozeModel):
    def __init__(self, config, config_global, logger):
        super(BiLSTMModel, self).__init__(config, config_global, logger)
        self.lstm_cell_size = self.config['lstm_cell_size']
        self.use_last_hidden = self.config.get('use_last_hidden', True)
        self.n_features = None

    def build(self, data, sess):
        self.n_features = data.n_features
        self.build_input(data, sess)
        self.initialize_weights()

        beginning_lstm = tf.nn.dropout(
            self.apply_lstm(
                self.embeddings_story_begin,
                self.input_story_begin,
                re_use_lstm=False
            ),
            self.dropout_keep_prob
        )

        ending_lstm = tf.nn.dropout(
            self.apply_lstm(
                self.embeddings_story_end,
                self.input_story_end,
                re_use_lstm=True
            ),
            self.dropout_keep_prob
        )

        concatenated = tf.concat(1, [beginning_lstm, ending_lstm, self.input_features])
        dense_1_out = tf.nn.relu(tf.nn.xw_plus_b(concatenated, self.dense_1_W, self.dense_1_b))
        dense_2_out = tf.nn.xw_plus_b(dense_1_out, self.dense_2_W, self.dense_2_b)

        self.create_outputs(dense_2_out)

    def initialize_weights(self):

        """
        Global initialization of weights for the representation layer

        """

        with tf.variable_scope('lstm_cell_fw'):
            self.lstm_cell_forward = rnn_cell.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)

        with tf.variable_scope('lstm_cell_bw'):
            self.lstm_cell_backward = rnn_cell.BasicLSTMCell(self.lstm_cell_size, state_is_tuple=True)

        # self.lstm_cell_size * 2 * 2:
        # the first multiplier "2" is because it is a bi-directional LSTM model (hence we have 2 LSTMs).
        #  The second "2" is because we feed the story context and an ending * separately *, thus
        # obtaining two outputs from the LSTM.
        self.dense_1_W = weight_variable('dense_1_W',
                                         [self.lstm_cell_size * 2 * 2 + self.n_features, self.lstm_cell_size])
        self.dense_1_b = bias_variable('dense_1_b', [self.lstm_cell_size])

        self.dense_2_W = weight_variable('dense_2_W', [self.lstm_cell_size, 2])
        self.dense_2_b = bias_variable('dense_2_b', [2])

    def apply_lstm(self, item, indices, re_use_lstm):
        """Creates a representation graph which retrieves a text item (represented by its word embeddings) and returns
        a vector-representation

        :param item: the text item. Can be question or (good/bad) answer
        :param sequence_length: maximum length of the text item
        :param re_use_lstm: should be False for the first call, True for al subsequent ones to get the same lstm variables
        :return: representation tensor
        """
        tensor_non_zero_token = non_zero_tokens(tf.to_float(indices))
        sequence_length = tf.to_int64(tf.reduce_sum(tensor_non_zero_token, 1))

        with tf.variable_scope('lstm', reuse=re_use_lstm):
            output, last_state = tf.nn.bidirectional_dynamic_rnn(
                self.lstm_cell_forward,
                self.lstm_cell_backward,
                item,
                dtype=tf.float32,
                sequence_length=sequence_length
            )

            if self.use_last_hidden:
                return tf.concat(1, [last_state[0][0], last_state[1][0]])
            else:
                return tf.reduce_max(tf.concat(2, output), [1], keep_dims=False)


component = BiLSTMModel
