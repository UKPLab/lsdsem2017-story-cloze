import sys
import logging
import yaml

import random
import numpy as np
import scipy as sp

import generate

def run(destination, config_path):
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

    random.seed(42)
    np.random.seed(42)
    sp.random.seed(42)

    with open(config_path, 'r') as config_file:
        config = yaml.load(config_file.read())
        generate.generate_negative_rocstory_endings(destination, **config)

if __name__ == '__main__':
    run(*sys.argv[1:])