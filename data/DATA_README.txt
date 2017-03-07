==================================================================
  LSDSem2017-UKP-ROCStories-StoryCloze
  v1.0
==================================================================

This data was used for our experiments described in the LSDSem 2017 paper:

"LSDSem 2017: Exploring Data Generation Methods for the Story Cloze Test"

Please cite the paper as:

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
  series    = {LSDSem '17},
  url       = {TBA}
}

------------------------------------------------------------------
 Description
------------------------------------------------------------------

The data is provided in three archives:
* `Data_for_neural_network.7z`: the training, development and test datasets with features, used to train and evaluate our proposed neural network system
* `ROCStories_generated_by_KDE_sampling.7z`: the output of our proposed KDE sampling method for generating ROCStories with wrong endings
* `Input_data_for_KDE_sampling.7z`: the necessary input data for running the KDE sampling method to generate the content of ROCStories_generated_by_KDE_sampling.7z. Contains segmentation and POS-tagging information for all of ROCStories and the two Story Cloze datasets.

------------------------------------------------------------------
 Data format
------------------------------------------------------------------

All CSV files use commas (',') as delimiters and double quotes ('"') as quotation marks.
Segmentation and POS-tags in Input_data_for_KDE_sampling.7z are stored in one text file per story context / story ending in CONLL 2009 format.

------------------------------------------------------------------
 Usage
------------------------------------------------------------------
The data is used as input for the neural network classification
system which is available at the following GitHub repository:

https://github.com/UKPLab/lsdsem2017-story-cloze

------------------------------------------------------------------
 License
------------------------------------------------------------------
* The data are licensed under CC-BY 4.0.
License details: http://creativecommons.org/licenses/by/4.0

* Please cite the LSDSem 2017 system description article if you use the data 
in any of your work.

* The original ROCStories and Story Cloze Test data keep the original license
(see http://cs.rochester.edu/nlp/rocstories/)
