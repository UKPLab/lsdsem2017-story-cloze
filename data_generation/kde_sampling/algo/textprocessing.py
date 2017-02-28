import re
import numpy as np

from string import punctuation

# snowball stopwords from http://snowball.tartarus.org/algorithms/english/stop.txt
_STOPWORDS = {'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as',
             'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't",
             'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down',
             'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't",
             'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself',
              'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's",
              'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off',
              'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same',
              "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that',
              "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they',
              "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up',
              'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's",
              'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with',
              "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
              'yourselves'}
_PATTERN_SUBJECT_APOSTROPHE = re.compile(r"\b(I|[Yy]ou|[Hh]e|[Ss]he|[Ii]t|[Ww]e|[Tt]hey)'\w+\b")
PATTERN_STRICT = re.compile(r"FW|(VB)\w?|NN\w?\w?")
PATTERN_LAX = re.compile(r"FW|(JJ|VB|RB)\w?|NN\w?\w?")
PATTERN_SUPER_LAX = re.compile(r".*")

_PRONOUNS_1ST_PERS_SG = {"I", "me", "my", "mine", "myself"}
_PRONOUNS_1ST_PERS_PL = {"we", "us", "our", "ours", "ourselves"}
_PRONOUNS_3RD_PERS_PL = {"they", "them", "their", "theirs", "themselves"}
PRONOUNS_3RD_PERS_SG_MALE = {"he", "him", "his", "himself"}
PRONOUNS_3RD_PERS_SG_FEMALE = {"she", "her", "hers", "herself"}
PRONOUNS_1ST_PERS = _PRONOUNS_1ST_PERS_SG | _PRONOUNS_1ST_PERS_PL
PRONOUNS_PL = _PRONOUNS_1ST_PERS_PL | _PRONOUNS_3RD_PERS_PL
PRONOUNS_SG = _PRONOUNS_1ST_PERS_SG | PRONOUNS_3RD_PERS_SG_MALE | PRONOUNS_3RD_PERS_SG_FEMALE
PRONOUNS = PRONOUNS_SG | PRONOUNS_SG
PRONOUNS_SUBJECT = {"I", "you", "he", "she", "we", "they"}   #ignore "it"

def remove_punctuation(s):
    return "".join(letter for letter in s if letter not in punctuation)


def get_sentence_from_conll(conll_document):
    """
    Given the path to a conll format file, rebuilds a (close approximation?) of the original sentence by appending the
    surface forms, separated by spaces.
    :para conll_document: the contents of a CONLL file, as a list of triples
    """
    return " ".join(tup[0] for tup in conll_document)


def get_pronoun_lemmas(conll_document):
    """
    Returns the list of lemmatized pronouns found in the given contents of a CONLL document (list of triples).
    :param conll_document: the contents of a CONLL file, as a list of triples
    :return:
    """
    pronouns = []
    for (form, lemma, pos) in conll_document:
        lemma_no_punct = remove_punctuation(lemma)
        if pos in {"PRP", "PRP$"} or lemma_no_punct in PRONOUNS:
            pronouns.append(lemma_no_punct)
        elif _PATTERN_SUBJECT_APOSTROPHE.match(lemma_no_punct):
            # identify and normalize occurrences of "I'd", "he's", "they've" and so on
            pronoun = _PATTERN_SUBJECT_APOSTROPHE.sub(r"\1", lemma_no_punct)
            pronouns.append(pronoun)
    return pronouns


def get_content_words(conll_document, pos_regex_pattern=PATTERN_STRICT):
    """
    Returns the set of content words given the contents of a CONLL file.
    :param conll_document: the contents of a CONLL file, as a list of triples
    :param pos_regex_pattern: every POS tag matched by this regex pattern will be considered a content word
    :return: the set of content words
    """
    content_words = set()
    for (form, lemma, pos) in conll_document:
        if pos_regex_pattern.match(pos):
            content_words.add(remove_punctuation(form))
    return content_words - _STOPWORDS


def avg_embedding_for_doc(document, embedding_io, content_word_func=lambda doc: get_content_words(doc)):
    """
    Returns the averaged embedding of the content words of a document.
    In case the given content word extraction function returns an empty set of words for a document, every word in the
    document is considered to be a content word as a fallback.
    :param document: some CONLL document content (list of triplets)
    :param embedding_io:
    :param content_word_func:
    :return:
    """
    content_words = content_word_func(document)
    if not content_words:
        content_words = get_content_words(document, PATTERN_SUPER_LAX)
    words_with_known_embeddings = [word for word in content_words if embedding_io.has_embedding(word)]
    word_embeddings = [embedding_io.get_embedding(word) for word in words_with_known_embeddings]
    embedding_vec = np.sum(word_embeddings, axis=0)
    embedding_vec /= np.linalg.norm(embedding_vec)
    return embedding_vec