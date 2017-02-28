import logging
import numpy as np
import progressbar as pb

from collections import OrderedDict


import algo.textprocessing as tp


class SimilarityMeasure:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_similarities(self, story_id):
        """
        Returns a numpy array of values within [-1, 1] denoting the similarity of a context document (identified by
        its id) to all ending documents passed in the initialize method.
        The order of the values in the returned array matches the iteration order of ending_documents (passed in the
        constructor). Therefore make sure you pass an OrderedDict!
        :param story_id: the story_id of the story for which the similarities are to be calculated
        :return:
        """
        raise NotImplementedError

    def get_similarity(self, context_document, ending_document):
        """
        Returns a single value from the interval [-1, 1] denoting the similarity of the two given documents.
        For large numbers of documents, get_similarities should be used instead.
        :param context_document:
        :param ending_document:
        :return:
        """
        raise NotImplementedError


class EmbeddingSimilarity(SimilarityMeasure):
    """
    Calculates the cosine similarity between content words of context and ending.
    """

    def __init__(self, context_documents, ending_documents, embedding_io):
        """

        :param context_documents:
        :param ending_documents: dict mapping from id to CONLL content. If you want to use get_similarities, make
        sure this is an OrderedDict!
        :param embedding_io:
        """
        super().__init__()
        assert type(ending_documents) is OrderedDict

        self.embedding_io = embedding_io

        self.logger.info("Calculating context embeddings...")
        self.context_embeddings = {}
        bar = pb.ProgressBar(max_value=len(context_documents))
        for doc_id, doc in bar(context_documents.items()):
            self.context_embeddings[doc_id] = tp.avg_embedding_for_doc(doc, self.embedding_io, lambda x: tp.get_content_words(x, tp.PATTERN_STRICT))

        self.logger.info("Calculating ending embeddings...")
        ending_embeddings = OrderedDict()
        bar = pb.ProgressBar(max_value=len(ending_documents))
        for doc_id, doc in bar(ending_documents.items()):
            ending_embeddings[doc_id] = tp.avg_embedding_for_doc(doc, self.embedding_io, lambda x: tp.get_content_words(x, tp.PATTERN_LAX))
        self.mat_endings = np.array(list(ending_embeddings.values()), dtype=np.dtype('f8'))

    def get_similarities(self, story_id):
        vec_context = self.context_embeddings[story_id]
        return np.dot(self.mat_endings, vec_context)

    def get_similarity(self, context_document, ending_document):
        context_embedding = tp.avg_embedding_for_doc(context_document, self.embedding_io,
                                                     lambda doc: tp.get_content_words(doc, tp.PATTERN_STRICT))
        ending_embedding = tp.avg_embedding_for_doc(ending_document, self.embedding_io,
                                                    lambda doc: tp.get_content_words(doc, tp.PATTERN_LAX))
        return np.dot(ending_embedding, context_embedding)


class PronounSimilarity(SimilarityMeasure):
    """
    Calculates similarity values based on the usage of pronouns in story context and endings.
    For a given pair of context and ending, the similarity is defined as follows: If either of the two contains
    pronouns denoting a 1st person speaker ("I", "we", "us", "myself", ...) the similarity is -1. Otherwise, the
    speaker in context and ending is very likely to be the same (2nd person speakers practically do not appear in
    ROCStories / StoryCloze datasets). In this case, similarity is computed by counting the occurrences of certain
    pronoun types (the number of singular pronouns, plural, male, female) for both context and ending, storing these
    counts in one (resp. two) vector(s) which is/are then normalized. Afterwards, cosine similarity is applied (dot
    product on the normalized vectors).
    """

    @staticmethod
    def _create_vec_pronoun(document, document_pronouns):
        """
        Creates a vector denoting the distribution of several types of pronouns in a given document. The resulting
        vector is normalized.
        vec[0]: singular pronouns (incl. NNP), vec[1]: plural pronouns (incl. NNPS), vec[2]: male 3rd person pronouns,
        vec[3]: female 3rd person pronouns
        :param document:
        :param document_pronouns:
        :return:
        """
        bins = [tp.PRONOUNS_SG, tp.PRONOUNS_PL, tp.PRONOUNS_3RD_PERS_SG_MALE, tp.PRONOUNS_3RD_PERS_SG_FEMALE]
        vec = []
        for b in bins:
            vec.append(sum([1 for pronoun in document_pronouns if pronoun in b]))
        vec[0] += sum([1 for triplet in document if triplet[2] == "NNP"])
        vec[1] += sum([1 for triplet in document if triplet[2] == "NNPS"])
        vec = np.array(vec, dtype=np.float16)
        vec += 0.05 * (2 * np.random.random_sample(len(vec)) - 1)    # add a little bit of jitter to create some variety in the vectors (in the hope that this reduces the reuse of sentences)
        if vec.any():
            vec /= np.linalg.norm(vec)
        return vec

    def __init__(self, context_documents, ending_documents):
        """

        :param context_documents:
        :param ending_documents: dict mapping from id to CONLL content. If you want to use get_similarities, make
        sure this is an OrderedDict!
        """
        super().__init__()
        assert type(ending_documents) is OrderedDict
        self.context_documents = context_documents

        list_of_ending_documents = ending_documents.values()

        # create a list containing the list of pronouns used in each i-th ending
        pronouns_in_endings = [tp.get_pronoun_lemmas(doc) for doc in list_of_ending_documents]

        # create a boolean vector where the i-th element denotes whether the i-th ending contains 1st person pronouns
        self.ith_ending_has_1st_pers_pronouns = np.array(
            [bool(set(list_of_used_pronouns) & tp.PRONOUNS_1ST_PERS) for list_of_used_pronouns in pronouns_in_endings],
            dtype=np.dtype(bool))

        # build pronoun vector for each ending -> "mat_endings"
        self.mat_endings = np.array(
            [self._create_vec_pronoun(doc, pron) for doc, pron in zip(list_of_ending_documents, pronouns_in_endings)],
            dtype=np.float16)

    def get_similarities(self, story_id):
        context_document = self.context_documents[story_id]

        # determine if the context contains 1st pers pronouns
        list_of_used_pronouns_in_context = tp.get_pronoun_lemmas(context_document)
        context_has_1st_pers_pronouns = bool(set(list_of_used_pronouns_in_context) & tp.PRONOUNS_1ST_PERS)

        # create a boolean vector whether the 1st person pronoun usage in the i-th ending mismatches that in the context
        vec_pronoun_mismatch = np.logical_xor(context_has_1st_pers_pronouns, self.ith_ending_has_1st_pers_pronouns)

        # reasoning: if an ending is written in 1st person but the context isn't (or the other way around), these
        # endings are very dissimilar, so their pairwise similarity will be set to -1 (lowest possible similarity score)
        similarities = vec_pronoun_mismatch * -1.0

        # build pronoun vector for the context
        vec_context = self._create_vec_pronoun(context_document, list_of_used_pronouns_in_context)

        # dot-product of "mat_endings" with "vec_context_pronoun" to compute the cosine similarity OF ALL pairs,
        # including those where the 1st person pronoun usage mismatches
        pronoun_similarities = np.dot(self.mat_endings, vec_context)

        # now invalidate those similarity values where 1st person pronoun usage mismatches by replacing each of the
        # affected values by 0
        fixed_pronoun_similarities = np.ma.filled(np.ma.masked_array(pronoun_similarities, mask=vec_pronoun_mismatch),
                                                  fill_value=0)

        # then add it to the similarities, leading to a result vector where values are either -1 (mismatch of 1st
        # person pronoun usage) or some cosine similarity value
        similarities += fixed_pronoun_similarities

        return similarities

    def get_similarity(self, context_document, ending_document):
        context_pronouns = tp.get_pronoun_lemmas(context_document)
        ending_pronouns = tp.get_pronoun_lemmas(ending_document)

        context_has_1st_pers_pronouns = bool(set(context_pronouns) & tp.PRONOUNS_1ST_PERS)
        ending_has_1st_pers_pronouns = bool(set(ending_pronouns) & tp.PRONOUNS_1ST_PERS)

        # one set has 1st pers. pronouns but the other doesn't -> highly different
        if context_has_1st_pers_pronouns != ending_has_1st_pers_pronouns:
            return -1
        else:
            # at this point, context/ending either both feature 1st or 3rd person subjects - grepping the data showed
            # that there are practically no stories using 2nd person
            vec_context = self._create_vec_pronoun(context_document, context_pronouns)
            vec_ending = self._create_vec_pronoun(ending_document, ending_pronouns)
            return np.dot(vec_ending, vec_context)


class LinearCombinationSimilarity(SimilarityMeasure):

    def __init__(self, measures, weights):
        """

        :param measures: list of similarity measures
        :param weights: list of weights, one corresponding to each measure
        """
        super().__init__()
        if not len(measures) == len(weights):
            raise ValueError("The number of measures and weights do not match.")

        self.measures = measures
        self.weights = weights

    def get_similarities(self, story_id):
        return sum([w * m.get_similarities(story_id) for m, w in zip(self.measures, self.weights)])

    def get_similarity(self, context_document, ending_document):
        return sum(
            [w * m.get_similarity(context_document, ending_document) for m, w in zip(self.measures, self.weights)])


def get_assignment_similarities(context_documents, ending_documents, similarity_measure):
    """
    Given two dicts of the form (key -> embedding) where both dicts share the same set of keys, returns the
    similarities between documents of the same key as a list.
    :param context_documents:
    :param ending_documents:
    :param similarity_measure:
    :return a list of the cosine similarities between each context and ending of the same id
    """
    similarities = []
    for doc_id in context_documents.keys():
        context = context_documents[doc_id]
        ending = ending_documents[doc_id]
        similarities.append(similarity_measure.get_similarity(context, ending))
    return similarities
