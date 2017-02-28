import importlib
import logging
import sys,os

import click
import numpy as np
import tensorflow as tf

from neural_network.config import load_config


@click.command()
@click.argument('config_file')
def run(config_file):
    """This program is the starting point for every neural_network. It pulls together the configuration and all necessary
    neural_network classes to load

    """
    config = load_config(config_file)
    config_global = config['global']

    # Allows the gpu to be used in parallel
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True

    # we allow to set the random seed in the config file to perform multiple subsequent runs with different
    # initialization values in order to compute an avg result score
    seed = config_global.get('random_seed', 1)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    with tf.Session(config=sess_config) as sess:
        # We are now fetching all relevant modules. It is strictly required that these module contain a variable named
        # 'component' that points to a class which inherits from neural_network.Data, neural_network.Model,
        # neural_network.Trainer or neural_network.Evaluator
        data_module = config['data-module']
        model_module = config['model-module']
        training_module = config['training-module']
        evaluation_module = config.get('evaluation-module', None)

        # The modules are now dynamically loaded
        DataClass = importlib.import_module(data_module).component
        ModelClass = importlib.import_module(model_module).component
        TrainingClass = importlib.import_module(training_module).component
        EvaluationClass = importlib.import_module(evaluation_module).component if evaluation_module else None

        # setup a logger
        logger = logging.getLogger('neural_network')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler_stdout = logging.StreamHandler(sys.stdout)
        handler_stdout.setLevel(config['logger']['level'])
        handler_stdout.setFormatter(formatter)
        logger.addHandler(handler_stdout)

        if 'path' in config['logger']:
            log_fn = os.path.abspath(config['logger']['path'])
            log_dir = os.path.dirname(log_fn)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            handler_file = logging.FileHandler(log_fn)
            handler_file.setLevel(config['logger']['level'])
            handler_file.setFormatter(formatter)
            logger.addHandler(handler_file)

        logger.setLevel(config['logger']['level'])

        # setup the data (validate, create generators, load data, or else)
        data = DataClass(config['data'], config_global, logger)
        logger.info('Setting up the data')
        data.setup()

        # build the model (e.g. compile it)
        model = ModelClass(config['model'], config_global, logger)
        logger.info('Building the model')
        model.build(data, sess)

        mode = config_global['mode']
        saver = tf.train.Saver()
        sess.run(tf.initialize_all_variables())

        if mode == "train":
            logger.info('Training mode')
            training = TrainingClass(config['training'], config_global, logger)

            # start the training process
            logger.info('Starting the training process')
            training.start(model, data, sess)

        elif mode == "predict":
            logger.info('Evaluation mode')
            assert evaluation_module is not None, "No eval module -- check the config file!"
            evaluation = EvaluationClass(config['evaluation'], config_global, logger)
            evaluation.start_prediction(model, data, sess, saver)

        else:
            logger.warning("Check the operation mode in the config file: %s" % mode)

        logger.info('DONE')


if __name__ == '__main__':
    run()
