import logging
import scipy as sp
import numpy as np
from scipy import stats

import algo.similarity_measure as similarity


class DatingService:
    """
    A 'dating service' in the sense that it identifies the best match for something given a large list of candidates,
    in our case an ending for a given story context.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_index_of_best_partner(self, vec_similarities, taboo_index):
        """
        Given a list of similarities (of what exactly is not relevant), returns the index of the "best" value. What
        is "best" is decided by inheriting classes.
        :param vec_similarities: a list of values each taken from the interval [-1, 1]
        :param taboo_index: if truthy, this index is taboo and cannot be returned by this method
        :return:
        """
        raise NotImplementedError


class KernelDensityDating(DatingService):
    def __init__(self, valid_set_context_documents, valid_set_wrong_ending_documents, similarity_measure):
        super().__init__()
        validation_similarities = similarity.get_assignment_similarities(valid_set_context_documents,
                                                                         valid_set_wrong_ending_documents,
                                                                         similarity_measure)
        self.validation_sim_kde_estimate = sp.stats.gaussian_kde(validation_similarities)

    def get_index_of_best_partner(self, vec_similarities, taboo_index):
        # try 20 times to sample an ending different from the original ending before giving up
        for reroll in reversed(range(20)):
            # we want to find an ending with a cosine sim close to this value
            target_sim = self.validation_sim_kde_estimate.resample(1)
            # find the index in the list of said closest ending
            best_ending_id_index = (np.abs(vec_similarities - target_sim)).argmin()

            # if index is different from the original ending or if we're giving up on rerolling
            if best_ending_id_index != taboo_index or not reroll:
                return best_ending_id_index


class ArgmaxDating(DatingService):
    def get_index_of_best_partner(self, vec_similarities, taboo_index):
        if not taboo_index:
            return np.argmax(vec_similarities)
        else:
            # if the true ending is part of the ending candidates, hide it from argmax by numpy masking
            mask = np.zeros(vec_similarities.shape)
            mask[taboo_index] = 1
            vec_similarities = np.ma.array(vec_similarities, mask=mask, dtype=np.dtype('f8'))
            return np.ma.argmax(vec_similarities)
