import logging
import progressbar as pb
from collections import Counter, OrderedDict

from inout.embeddings import EmbeddingsFromMemory
from algo.datingservice import KernelDensityDating, ArgmaxDating
from algo.similarity_measure import EmbeddingSimilarity, PronounSimilarity, LinearCombinationSimilarity


def find_best_endings(context_documents, ending_documents, **kwargs):
    def prepare_similarity_measures(context_documents, ending_documents, embedding_io, use_pronoun_similarity):
        use_pronoun_similarity = kwargs["use_pronoun_similarity"]
        if use_pronoun_similarity:
            sim_pronouns = PronounSimilarity(context_documents, ending_documents)
            sim_embeddings = EmbeddingSimilarity(context_documents, ending_documents, embedding_io)
            sim_pronoun_weight = kwargs["pronoun_similarity_weight"]
            sim_embedding_weight = 1 - sim_pronoun_weight
            return LinearCombinationSimilarity([sim_embeddings, sim_pronouns],
                                               [sim_embedding_weight, sim_pronoun_weight])
        else:
            return EmbeddingSimilarity(context_documents, ending_documents, embedding_io)

    logger = logging.getLogger(__name__)

    logger.info("Loading embeddings...")
    embedding_io = EmbeddingsFromMemory(kwargs["embeddings_path"], kwargs["embeddings_uncased"])

    # sort the ending documents dictionary list indices will be used in the following
    ending_documents = OrderedDict(sorted(ending_documents.items()))
    list_of_ending_ids = list(ending_documents.keys())

    # define the similarity measure
    similarity_measure = prepare_similarity_measures(context_documents, ending_documents, embedding_io,
                                                     kwargs["use_pronoun_similarity"])

    # define the dating service
    use_kde = kwargs["use_kde"]
    if use_kde:
        valid_set_context_documents = kwargs["valid_context_documents"]
        valid_set_wrong_ending_documents = kwargs["valid_wrong_ending_documents"]
        logger.info("Preparing validation set for KDE...")
        valid_set_similarity_measure = prepare_similarity_measures(valid_set_context_documents,
                                                                   valid_set_wrong_ending_documents, embedding_io,
                                                                   kwargs["use_pronoun_similarity"])

        dating_service = KernelDensityDating(valid_set_context_documents, valid_set_wrong_ending_documents,
                                             valid_set_similarity_measure)
    else:
        dating_service = ArgmaxDating()

    # find a similar ending to each context, store this in a dict mapping from story context id to ending id
    logger.info("Finding an ending for each story context...")
    best_ending_id_for_story_id = {}
    bar = pb.ProgressBar(max_value=len(context_documents))
    for story_id in bar(context_documents.keys()):
        similarities = similarity_measure.get_similarities(story_id)

        # detect whether the true ending of this story is part of the candidate endings (i.e. same story_id present
        # in the ending_documents dict)
        true_ending_index = list_of_ending_ids.index(story_id) if story_id in ending_documents else None
        index_of_best_ending_id = dating_service.get_index_of_best_partner(similarities, true_ending_index)
        best_ending_id_for_story_id[story_id] = list_of_ending_ids[index_of_best_ending_id]

    # track which ending was used how often
    ending_usage_counts = Counter(best_ending_id_for_story_id.values())
    return best_ending_id_for_story_id, ending_usage_counts
