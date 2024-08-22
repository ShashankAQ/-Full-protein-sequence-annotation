## Using Deep Learning to Annotate the Protein Universe

Residual Network for Pfam Protein Sequence Annotation

Model Inspired by "Using Deep Learning to Annotate the Protein Universe"

Paper: https://www.biorxiv.org/content/10.1101/626507v2

Trained Model: https://console.cloud.google.com/storage/browser/dbtx-storage/Deeplearning/saved_model/?project=dbtx-pipeline

Data Source: https://console.cloud.google.com/storage/browser/brain-genomics-public/research/proteins/pfam/random_split

### Training

The model was trained using Pytorch Resnet18 with 1 channel and 64 filters from torch vision models on the pfam seed sequences (1086741 Sequences) with 17931 Families in Pfam 32.0 Release. The model was trained with minibatch 100 for 5 epochs, where each epoch switches between the train splits in the data

Training Accuracy: 98.932 % 

### Testing 
The model was then saved and tested on 126171 sequences from Pfam (Full), where the sequences have not been seen at all during training. 

Testing Accuracy: 95.407 %

### Make Predictions:

    # Make Inference on an entire Fasta File 
    python3 inference.py -m fasta -i <Path to Protein Fasta> -mpath ../saved_models/resnet_pfam_final_8.mdl
    # Make Inference from STDIN
    python3 inference.py -m string -i AAQFVAEHGDQVCPAKWTPGAETIVPSL -mpath ../saved_models/resnet_pfam_final_8.mdl
-------------------------
# Google Dataset Description
## Problem description 
This directory contains data to train a model to predict the function of protein domains, based
on the PFam dataset.

Domains are functional sub-parts of proteins; much like images in ImageNet are pre segmented to 
contain exactly one object class, this data is presegmented to contain exactly and only one
domain.

The purpose of the dataset is to repose the PFam seed dataset as a multiclass classification 
machine learning task.
 
The task is: given the amino acid sequence of the protein domain, predict which class it belongs
to. There are about 1 million training examples, and 18,000 output classes.

## Data structure
This data is more completely described by the publication "Can Deep Learning
Classify the Protein Universe", Bileschi et al.

### Data split and layout
The approach used to partition the data into training/dev/testing folds is a random split.

- Training data should be used to train your models.
- Dev (development) data should be used in a close validation loop (maybe
  for hyperparameter tuning or model validation).
- Test data should be reserved for much less frequent evaluations - this
  helps avoid overfitting on your test data, as it should only be used
  infrequently.

### File content
Each fold (train, dev, test) has a number of files in it. Each of those files
contains csv on each line, which has the following fields:






