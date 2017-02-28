from __future__ import division

import numpy as np
import tensorflow as tf
import neural_network
import logging
import os, sys
import csv

logger = logging.getLogger('evaluation')

class ScEvaluation(neural_network.Evaluation):
    def __init__(self, config, config_global, logger):
        super(ScEvaluation, self).__init__(config, config_global, logger)

    def start_prediction(self, model, data, sess, saver):

        assert 'checkpoint_dir' in self.config_global
        ckpt = tf.train.get_checkpoint_state(self.config_global['checkpoint_dir'])
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            self.logger.warning("No checkpoint found!")
            sys.exit()

        # Run the model to get predictions
        self.logger.info('Creating test data')
        self.data = []  # a list of beginning-vecs, ending-1-vecs, ending-2-vecs, label, features
        sentence_len = self.config_global["sentence_length"]
        for story in data.dataset.test.stories:
            beginning_vecs = data.get_item_vector(story.sentences, sentence_len * 4)
            ending_1_vecs = data.get_item_vector([story.potential_endings[0]], sentence_len)
            ending_1_features = np.array(story.potential_endings[0].metadata.get('feature_values', []))
            ending_2_vecs = data.get_item_vector([story.potential_endings[1]], sentence_len)
            ending_2_features = np.array(story.potential_endings[1].metadata.get('feature_values', []))
            label = story.correct_endings[0]
            story_id = story.id
            # TODO: add the story id part
            self.data.append(
                (beginning_vecs, ending_1_vecs, ending_1_features, ending_2_vecs, ending_2_features, label, story_id)
            )

        gold = []
        pred = []

        for idx, (beginning_vecs,
                  ending_1_vecs, ending_1_features,
                  ending_2_vecs, ending_2_features,
                  label, story_id) in enumerate(self.data, start=1):

            predict, = sess.run([model.predict], feed_dict={
                model.input_story_begin: [beginning_vecs] * 2,
                model.input_story_end: [ending_1_vecs, ending_2_vecs],
                model.input_features: [ending_1_features, ending_2_features],
                model.dropout_keep_prob: 1.0})

            label_prediction = 0 if predict[0][0] > predict[1][0] else 1
            gold.append(label)
            pred.append((story_id, label_prediction))

        num_stories = len(gold)
        assert num_stories == len(pred)
        output_fn = os.path.join(self.config_global['checkpoint_dir'], 'answer.txt')
        correct = 0
        header = ["InputStoryid","AnswerRightEnding"]
        with open(output_fn, "w") as out:
            writer = csv.writer(out, delimiter=',', quotechar='"')
            writer.writerow(header)
            for idx, prediction in enumerate(pred):
                story_id, pred_label = prediction
                writer.writerow([story_id] + [pred_label+1])

                if pred_label == gold[idx]:
                    correct +=1

        self.logger.info("Done. Result: %0.2f" % (correct/num_stories * 100))

component = ScEvaluation
