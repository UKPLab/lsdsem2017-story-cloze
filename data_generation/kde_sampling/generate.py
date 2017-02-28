import logging

from inout.rocstories import write_rocstories_with_negative_endings
from inout.conll import read_conll_documents
from algo.find_endings import find_best_endings
import algo.ending_refiner as refinery


def generate_negative_rocstory_endings(dest_path, **config):
    """
    Generates a wrong ending to each ROCStory and stores the result in the same format as the StoryCloze test /
    validation CSV files.
    Given a set of preprocessed ROCStory story contexts and a set of preprocessed sentences, this method first extracts
    content words from story contexts and endings. Then, an averaged word embedding is created for each story context
    and ending. For each story context, the "closest" ending is determined in terms of cosine similarity of the word
    embeddings. Such an ending is then considered to be a good "wrong ending" to a ROCStory (resp. its story context).
    In case this method is called with the set of ROCStories endings as candidate endings, comparisons between id's
    (file names) of story context and endings ensure that a story context cannot be assigned its original ending.
    :param dest_path: destination path for the ROCStories CSV with wrong endings
    :param config:
    :return:
    """
    logger = logging.getLogger(__name__)

    logger.info("Reading ROCStories context and candidate ending CONLL files...")
    context_documents = read_conll_documents(config["rocstories_context_conll_path"])
    ending_documents = read_conll_documents(config["endings_conll_path"])

    if config["use_kde"]:
        logger.info("Reading StoryCloze validation set context and wrong endings CONLL files...")
        valid_context_documents = read_conll_documents(config["storycloze_valid_context_conll_path"])
        valid_wrong_ending_documents = read_conll_documents(config["storycloze_valid_wrong_endings_conll_path"])

        logger.info("Starting generation process with KDE sampling...")
        config.update({"valid_context_documents": valid_context_documents,
                       "valid_wrong_ending_documents": valid_wrong_ending_documents})
        best_ending_id_for_story_id, ending_usage_counts = find_best_endings(context_documents, ending_documents,
                                                                             **config)
    else:
        logger.info("Starting ending generation process with argmax selection...")
        best_ending_id_for_story_id, ending_usage_counts = find_best_endings(context_documents, ending_documents,
                                                                             **config)

    logger.info("Refining the selected endings...")
    refiner = refinery.ProperNounRefiner() if config["refine_endings"] else refinery.NopEndingRefiner()
    wrong_ending_for_story_id = {story_id: refiner.refine(context_documents[story_id], ending_documents[ending_id]) for
                                 story_id, ending_id in best_ending_id_for_story_id.items()}

    logger.info("Complete. Writing output ROCStories file with wrong endings...")
    write_rocstories_with_negative_endings(dest_path, config["rocstories_path"], wrong_ending_for_story_id)

    stats = "\n".join(str(t) for t in ending_usage_counts.most_common(100))
    logger.info("Done. The 100 most used endings were (ending, count):\n{}".format(stats))
