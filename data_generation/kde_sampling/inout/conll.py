from os import path, listdir
from collections import OrderedDict

import progressbar as pb

CONLL_FILE_EXT = ".conll"


def read_conll_file(conll_file_path):
    """
    Reads a CONLL-2009 file and returns its contents as a list of (form, lemma, pos) tuples.
    :param conll_file_path:
    :return:
    """
    tuples = []
    with open(conll_file_path, 'r') as file:
        for line in file:
            if line.strip():
                rows = line.split("\t")
                form = rows[1]
                lemma = rows[2]
                pos = rows[4]
                tuples.append((form, lemma, pos))
    return tuples


def read_conll_documents(conll_dir_path):
    conll_filenames = [path.join(conll_dir_path, fn) for fn in listdir(conll_dir_path) if not path.isfile(fn)]
    conll_documents_by_id = OrderedDict()

    bar = pb.ProgressBar(max_value=len(conll_filenames))
    for filename in bar(conll_filenames):
        id = path.splitext(path.basename(filename))[0]
        file_contents = read_conll_file(filename)
        conll_documents_by_id[id] = file_contents
    return conll_documents_by_id