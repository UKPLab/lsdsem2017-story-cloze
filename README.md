# LSDSem 2017: Exploring Data Generation Methods for the Story Cloze Test

This repository contains the code needed to reproduce the results reported in Bugert et al., *LSDSem 2017: Exploring Data Generation Methods for the Story Cloze Test*.

Please cite the paper as:

```
@InProceedings{lsdsem2017-bugert-exploring,
  author    = {Bugert, Michael and Puzikov, Yevgeniy and Rücklé, Andreas and
               Eckle-Kohler, Judith and Martin, Teresa and Martinez Camara, Eugenio and
               Sorokin, Daniil and Peyrard, Maxime and Gurevych, Iryna},
  title     = {{LSDSem 2017: Exploring Data Generation Methods for the Story Cloze Test}},
  booktitle = {Proceedings of the 2nd Workshop on Linking Models of Lexical, Sentential and Discourse-level Semantics (LSDSem)},
  month     = {April},
  year      = {2017},
  address   = {Valencia, Spain},
  publisher = {Association for Computational Linguistics},
  pages     = {(to appear)},
  url       = {TBA}
}
```

> **Abstract:** The Story Cloze test is a recent effort in providing a common test scenario for text understanding systems. 
As part of the LSDSem 2017 shared task, we present a system based on a deep learning architecture combined with a rich set of manually-crafted linguistic features. The system outperforms all known baselines for the task, suggesting that the chosen approach is promising. We additionally present two methods for generating further training data based on stories from the ROCStories corpus. 

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/

Don't hesitate to contact us if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and 
is published for the sole purpose of giving additional 
background details on the respective publication. 

## Project structure

* `config_files/` -- this folder contains configuration files
* `data/` -- this folder contains system input files
* `data_generation/` -- the scripts for generating training data ("Shuffling" and "KDE Sampling" methods)
* `neural_network/` -- all the scripts necessary for running the experiments
* `requirements.txt` -- a text file with the names of the required Python modules

## Requirements

* 64-bit Linux versions (not tested on other platforms)
* Python 2.7 or Python 3.5
* Python modules in the `requirements.txt` file
* [TensorFlow] [tensorflow] (tested on v0.11.0)
* Suitable word embeddings in **text format** (e.g. [glove.6B.zip] [glove_embed])

## Running the experiments
    
The general procedure for running any experiment is:
* Fill out all relevant configuration options in the config file
* Run the system in training or prediction mode: 

    ```$python run_experiment.py <config_file>```

* Run a significance test: 
    
    ```$python run_significance_test.py <pred_1> <pred_2> <gold_answers>```

Here, `<pred_1>` and `<pred_2>` are predictions of the system (two different models)
and `<gold_answers>` is the file with correct predictions.
All three files should have the official CSV submission format (see [here] [lsdsem17-eval]).

The `run_significance_test.py` script computes the [McNemar's test] [mcnemar-wiki] 
(as implemented in the [Statsmodels] [mcnemar-statsmodels] Python library) and prints the result 
 to standard output.  

### Expected results
 
Each configuration file has a `global` part with a `checkpoint_dir` field.
After finishing the **training** procedure, you should expect the model to be saved into the folder specified in this field.

Also, the `logging` part of the configuration file specifies the path where you can save the log file (be it a log of the training or prediction procedure).

When performing **prediction** on test data, the folder specified by `checkpoint_dir` will also contain `answer.txt` file with predicted values in the submission format.

The best model (BILSTM-VF) achieved 71.7% accuracy on the official test data.
The model was trained with the following parameter values:
    
  - Sentence length: 20
  - Embeddings: glove.6B.100d (lowercased data)
  - BiLSTM-VF model (trainable embeddings, cell size 141, use_last_hidden=true)
  - Optimizer: Adam
  - Initial learning rate: 0.0001
  - Batch size: 40
  - Dropout: 0.3
  - Num_epocs: 30


### Parameter description
The parameters are documented in any of the configuration files in the `config_files` folder. 

Please let us know if you have any questions regarding the parameters' meaning.

### Feature description

The system can be trained with additional lexical features or without them (using only word embedding vectors).

In the `config_files/` folder you can find four subfolders with sample configuration files:

  - bilstm-t
  - bilstm-tf
  - bilstm-v
  - bilstm-vf

Configuration files are customizable. You can create your own ones and experiment with various values for the fields. In order to make it easier to reproduce the results from the paper, we defined two types of configuration files, `train.yaml` and `predict.yaml`, which are used for training and evaluating models, respectively. 
 
Note that the main difference between the configuration files in different folders is the value of the `data-module` field. The following subsections describe how to prepare a dataset in the suitable format.

#### Type 1: word embeddings only 

* Prepare three (train, dev, test) CSV files with the following header:

    `story_id, sent1, sent2, sent3, sent4, ending1, ending2, label`

* Folders `bilstm-t/` and `bilstm-v/` contain sample configuration files which use only word embeddings. Note that the `data-module` field is set to `csv_reader-t` and `csv_reader-v`, respectively. You can change fields other than `data-module` as you see fit (e.g., for hyperparameter optimization)

#### Type 2: word embeddings and lexical features

We used the [DKPro TC](https://dkpro.github.io/dkpro-tc/) framework to extract features and integrate them into our deep learning architecture.
You can use any features you want, but they should comply with the following naming conventions:        
    
   - if the feature is defined for one ending, its name in the CSV header file should contain either "E1" (the first ending) or "E2" (the second ending): e.g., "SentimentE1", "SentimentE2".
   - if the feature is defined for both endings, then its name in the CSV header file **should not** contain "E1" or "E2" 
    ("LengthDiff", "BigramOverlapDiff", etc.)
     
If you decide to incorporate features into the network, the general procedure to create an input for the system is as follows:

* Prepare three (train, dev, test) CSV files (or use ours from the `data/` folder) with the following header:
    
    `story_id, sent1, sent2, sent3, sent4, ending1, ending2, label, feature1, feature2, ..., featureN`

* Folders `bilstm-tf/` and `bilstm-vf/` contain sample configuration files which use both word embeddings and extracted lexical features. Note that the `data-module` field is set to `csv_reader-tf` and `csv_reader-vf`, respectively. 
   You can change fields other than `data-module` as you see fit (e.g., for hyperparameter optimization)


## Data
The `data/` folder contains:
* `Data_for_neural_network.7z`: the training, development and test datasets with features, which we used for our experiments
* `ROCStories_generated_by_KDE_sampling.7z`: the output of our proposed KDE sampling method for generating ROCStories with wrong endings
* `Input_data_for_KDE_sampling.7z`: the necessary input data for running said method (see below)

## Data generation
### Shuffling
Running ```$python shuffling.py <src> <dest>``` with a ROCStories CSV file as `<src>` will create a second CSV file at `<dest>`. This file will contain two lines for each ROCStory: the original (correct) story and a shuffled (wrong) story. Correct stories are marked with value 1 in the ```label``` column.

### KDE Sampling
* Extract the `Input_data_for_KDE_sampling.7z` archive from the `data/` folder.
* Specify the paths to the individual input files and pretrained embeddings (we used [glove.6B.zip] [glove_embed]) in the `config.yaml` file.
* Run the data generation via:
    ```$python3 run.py <dest_csv> <config.yaml>```
* The output file `<dest_csv>` will be a CSV file in the format of the Story Cloze validation/test sets, containing ROCStories with generated wrong endings.
 

[//]: # (Reference links. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)
   
   [orig-paper-a-corpus]: <http://www.aclweb.org/anthology/N/N16/N16-1098.pdf>
   [numpy]: <http://www.numpy.org/>
   [tensorflow]: <https://www.tensorflow.org/>
   [glove_embed]: <http://nlp.stanford.edu/data/glove.6B.zip>
   [lsdsem17-eval]: <https://competitions.codalab.org/competitions/15333#learn_the_details-evaluation>
   [mcnemar-wiki]: <https://en.wikipedia.org/wiki/McNemar's_test>
   [mcnemar-statsmodels]: <http://www.statsmodels.org/stable/generated/statsmodels.stats.contingency_tables.mcnemar.html#statsmodels.stats.contingency_tables.mcnemar>
   
