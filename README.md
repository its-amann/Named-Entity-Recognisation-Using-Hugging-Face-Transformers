# Named Entity Recognition (NER) with RoBERTa

This README provides a detailed explanation of the code for a Named Entity Recognition (NER) model using the RoBERTa pre-trained language model.

## Installation
This section explains the necessary library installations.


`!pip install transformers datasets evaluate`

`!pip install seqeval`

**Why use these commands?**

*   **`!pip install transformers datasets evaluate`**: 
    *   **Motive:** This command installs core libraries required for the project. The project relies on using a pretrained language model (RoBERTa). Thus, a library to import the models and a tool to get the needed dataset is required.
    *   **Logic:**
        *   `transformers`: The `transformers` library from Hugging Face provides access to pre-trained models like RoBERTa and tools for working with them.
        *   `datasets`: The `datasets` library helps to download, manage, and process various text datasets, like the CoNLL2003 data we are using here.
        *   `evaluate`: The `evaluate` library provides tools for evaluating our model’s performance with different metrics.
*   **`!pip install seqeval`**:
    *   **Motive**: The `seqeval` library is used to evaluate performance metrics.
    *   **Logic**: seqeval library provides metrics which will be used to assess the accuracy of the prediction of the trained model.

## Imports
This section shows the libraries imported in order to proceed with data processing and model development.
```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
import datetime
import pathlib
import io
import os
import re
import string
import evaluate
import time
from numpy import random
import gensim.downloader as api
from PIL import Image
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense,Flatten,InputLayer,BatchNormalization,Dropout,Input,LayerNormalization
from tensorflow.keras.losses import BinaryCrossentropy,CategoricalCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.metrics import Accuracy,TopKCategoricalAccuracy, CategoricalAccuracy, SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam
from google.colab import drive
from google.colab import files
from datasets import load_dataset
from transformers import (BertTokenizerFast,TFBertTokenizer,BertTokenizer,RobertaTokenizerFast,DataCollatorForTokenClassification,
                          DataCollatorWithPadding,TFRobertaForSequenceClassification,TFBertForSequenceClassification,
                          TFBertModel,create_optimizer,TFRobertaForTokenClassification,TFAutoModelForTokenClassification,)
```

**Why use these imports?**
The imports are used as tools in the project and are called and utilized by other functions.

*   **Tensorflow Related:** 
    *   **Motive**: These tools are essential for deep learning tasks, allowing building, training and testing neural networks. 
    *   **Logic**:
        *   `tensorflow`: A deep learning library required for constructing the neural network model.
        *   `tensorflow.keras` (especially layers, losses, metrics, optimizers):  Used to build and train the model as well as calculating performance metrics of the model.
*   **Data Manipulation:**
    *   **Motive**: Libraries are used for numerical manipulation of data for preparing the input to be fitted in the deep learning models.
    *   **Logic**:
        *   `numpy`: Used to manipulate numerical arrays, necessary for data processing
        *   `pandas`:  To prepare structured data (in tabular format)
*   **Visualization and Metrics:**
    *   **Motive**: Visualization libraries are used to depict performance while `sklearn.metrics` allows to evaluate the performance.
    *   **Logic**:
        *   `matplotlib.pyplot`: Used for plotting graphs to better understand the behaviour of the training process.
        *   `sklearn.metrics`: Provides various metrics (e.g., confusion matrices, ROC curves) to evaluate our model.
        *   `seaborn`: Used for improved visualization
*   **Text Processing and Transformers:**
    *   **Motive**: These libraries allow us to get a pre-trained model to train on the NER task and prepare it according to the dataset being used.
    *   **Logic**:
        *   `datasets.load_dataset`: Loads datasets.
        *   `transformers` (various classes such as `RobertaTokenizerFast`, `TFRobertaForTokenClassification` and DataCollator related classes): Used to load RoBERTa model, the tokenizer and data collators needed to train the data on our models.

## Hyperparameters

```python
BATCH_SIZE=16
NUM_EPOCHS=2
```
**Why use these variables?**
These variables set the parameters for the training process of the model.
*  `BATCH_SIZE`: Defines the amount of examples processed together in one step of training.
*  `NUM_EPOCHS`: Defines the number of times that the model has to train on all the training examples provided to the dataset.
## Data Preparation
This section prepares the dataset to be fitted in the model.
### Loading Dataset
```python
dataset = load_dataset("conll2003")
```
**Why use this command?**
This command loads the CoNLL2003 dataset.
*   **Motive**: Provides a standard dataset for NER. The model will learn from the examples of this dataset and be able to extract entities from any type of text.
*   **Logic**: 
    *   `load_dataset("conll2003")` function will load CoNLL2003 dataset directly from Hugging Face library.
### Dataset Exploration
```python
dataset
```
**Why use this command?**
 This code is used to display the structure of the loaded dataset.
*   **Motive**: Provides insights on the data’s structure (like splits, features).
*   **Logic**: It helps to have a better idea of the data that is going to be processed further.

```python
dataset['train'][20]
```
**Why use this command?**
Displays an example from the training set, providing valuable info about it.
*   **Motive**:  Inspects the data, understand its different fields.
*   **Logic**: This is used to have a deeper look into how data is structured.

### Tokenization and Label Alignment

```python
model_id="roberta-base"
tokenizer=RobertaTokenizerFast.from_pretrained(model_id,add_prefix_space=True)
```
**Why use this code?**
This loads a tokenizer from a pre-trained model for transforming sentences into a series of tokens that can be used by the model.
*   **Motive**:  Prepares the tokenizer to create a sequence of tokens that the model is familiar with.
*  **Logic**: 
 *   `model_id="roberta-base"`: Specifies the ID of the model to load.
*  `RobertaTokenizerFast.from_pretrained`: function loads the tokenizer related to the `roberta-base` model id.
* `add_prefix_space`: sets the parameter so the tokenizer can handle any given input.
```python
inputs = tokenizer(dataset["train"][20]["tokens"], is_split_into_words=True,)
inputs.tokens()
```
**Why use this code?**
The tokens created from using the tokenizer.
* **Motive:** View the effect of using a tokenizer on a text.
* **Logic**:
 `inputs.tokens()` shows how the tokenizer process an specific example with it token ids.
```python
print(dataset['train'][20])
```
**Why use this code?**
 Prints the original data.
* **Motive:** To have the original text without any preprocessing.
* **Logic**: 
 `print(dataset['train'][20])` shows the token of the specified example
```python
print(inputs.word_ids())
print(dataset['train'][20]['ner_tags'])
```
**Why use this code?**
 Prints the tokenized word ids alongside the true labels.
* **Motive:** Shows how word id mapping happens from the tokenizer to the dataset's NER labels
* **Logic**:
   *   `inputs.word_ids()`: Provides ids to represent the tokens created, mapping each to an index, and assigning the special tokens a None value.
  *  `dataset['train'][20]['ner_tags']`: This shows how the tokens in the current example are labelled by human annotators.
```python
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels
```
**Why use this function?**
 This custom function ensures the label sequence matches the tokenized sequence.
*   **Motive**: Tokenizers might break up words; a method to align those with the given labels is needed, ensuring each sub-token is associated to the correct named entity label or padding (-100).
*   **Logic**:
    *   Iterates over tokenized sequences.
    *   If new word, assigns label.
    *   -100 if token is from tokenizer, padding labels is a necessity
    *  If B-XXX we change it to I-XXX if multiple sub-tokens exist within the word.
```python
labels = dataset["train"][20]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))
```
**Why use this code?**
  This part shows the outcome of the `align_labels_with_tokens` function.
* **Motive:** This verifies the performance of the  alignment.
*   **Logic**: Outputs the original labels, alongside with the aligned ones, showcasing the mapping.

```python
def tokenizer_function(dataset):
  out=tokenizer(dataset["tokens"],truncation=True,is_split_into_words=True,)
  out['labels']=align_labels_with_tokens(dataset["ner_tags"],out.word_ids())
  return out
```
**Why use this function?**
 Tokenize and align labels to datasets.
*   **Motive**: This function encapsulates the preprocessing that's going to be needed in the tokenized dataset, such as tokenization, truncation, and label alignment
*  **Logic**: Applies tokenization, handling truncation, and alignment on the provided dataset by calling previous functions and combining them, resulting in the complete input structure.

```python
tokenized_dataset=dataset.map(tokenizer_function,remove_columns=['id','tokens','pos_tags','chunk_tags','ner_tags',])
```
**Why use this function?**
 Applies tokenization on entire data splits.
*   **Motive**: This operation preprocesses the dataset by applying tokenization, padding and aligning each example in the train, validation and test splits
*  **Logic**: This applies the  `tokenizer_function` on every data split of the dataset (`train`, `validation`, `test`), so the models can fit into it during the training phase, and also dropping the original columns which are no longer required (`'id','tokens','pos_tags','chunk_tags','ner_tags',`).
```python
tokenized_dataset
```
**Why use this code?**
 Verifies the transformation of datasets.
* **Motive:** See if the output of the previous function (`tokenized_dataset`) aligns to the expected behaviour.
* **Logic**: This command will display the transformed dataset structure including input\_ids, attention\_mask and labels, which are compatible with transformer models.

```python
tokenized_dataset['train'][20]
```
**Why use this code?**
 This function displays a processed example of data.
* **Motive:** Verifies individual data example processing.
*   **Logic**: Verifies data with tokens, attention masks and the labels aligned correctly.

### Data Collator
```python
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, return_tensors="tf"
)
```
**Why use this command?**
 This section sets a data collator for efficient batching of processed data.
*   **Motive**: For efficient batch processing as different sequences can have different lengths, it's necessary to pad the data.
*   **Logic**: The `DataCollatorForTokenClassification` pads examples of a dataset, including labels as well, according to the maximum length in a batch and it uses the tokenizer defined earlier to accomplish this task.

### TensorFlow Dataset Preparation

```python
tf_train_dataset = tokenized_dataset["train"].to_tf_dataset(
    collate_fn=data_collator,
    shuffle=True,
    batch_size=BATCH_SIZE,
)
```
**Why use this code?**
 Create TensorFlow-friendly dataset.
*   **Motive**: Convert our data format to one which can be understood by Tensorflow models and is usable by them for training.
*  **Logic**: Converts the tokenized train dataset to a `tf.data.Dataset` using `to_tf_dataset`. Padding, shuffling and batching is done through `DataCollatorForTokenClassification`.

```python
tf_val_dataset = tokenized_dataset["validation"].to_tf_dataset(
    collate_fn=data_collator,
    shuffle=False,
    batch_size=BATCH_SIZE,
)
```
**Why use this code?**
 Create validation Tensorflow datasets
*   **Motive**: Same as above, but for the validation data splits, to use during model training.
*   **Logic**: Converts tokenized validation data to TensorFlow datasets, while keeping batch size and using same padding function.
```python
for i in tf_train_dataset.take(1):
  print(i)
```
**Why use this code?**
 This function views a sample batch of data
*   **Motive**: Inspects a sample of data to verify its correct shape.
* **Logic**:
  This iteration of the first element `i` of the `tf_train_dataset` dataset helps to verify batching, padding, labels and attention masking by displaying one sample batch of data.

## Modeling

```python
model=TFRobertaForTokenClassification.from_pretrained(
    model_id,
    num_labels=9,
)
```
**Why use this command?**
 Loading the pre-trained RoBERTa model for token classification.
*   **Motive**: Uses a model already pre-trained in order to solve the problem of sequence labeling
* **Logic**:
    *   `TFRobertaForTokenClassification.from_pretrained`:  Loads a pre-trained RoBERTa model for token classification from the `transformers` library by using the pre-defined model id.
    *   `num_labels=9`: Indicates how many categories will exist in the final classification, setting the output dimension of the last layer.
```python
model.summary()
```
**Why use this command?**
 Displays a summary of model architecture
*   **Motive**: View the model architecture to understand trainable parameters of the model.
* **Logic**: `model.summary()` method prints the layers, output shapes, and parameter counts of the loaded RoBERTa. This verifies that the model has been properly loaded and initialized.

## Training

```python
batches_per_epoch = len(tokenized_dataset["train"]) // BATCH_SIZE
total_train_steps = int(batches_per_epoch*NUM_EPOCHS)
```
**Why use these commands?**
Calculates the training steps
*   **Motive**: Set parameters related to training steps for the optimizer
*   **Logic**: Calculates `batches_per_epoch` and `total_train_steps` from the data and hyperparameter variables which will be used to schedule the learning rate of the optimizer.

```python
optimizer, schedule = create_optimizer(init_lr=2e-5,num_warmup_steps=0, num_train_steps=total_train_steps,)
```
**Why use this command?**
 Creates an optimizer using a defined schedule
*  **Motive:** Sets the optimizer with the correct learning rate as well as applying the schedule, learning rate will be dynamically changed
*   **Logic**: It initializes the optimizer (`Adam`) and learning rate scheduler used during the training of the RoBERTa model, learning rate has a warm up step that is not currently being used here.

```python
model.compile(optimizer=optimizer,)
```
**Why use this command?**
 Configures model with optimizer.
*   **Motive**: Makes the model use the given optimizer and use default loss, which will use loss as defined internally by the `TFRobertaForTokenClassification` class.
*   **Logic**: Model configuration allows it to start the training process, using the defined optimizer and will automatically calculate the loss during training as well.

```python
history=model.fit(
    tf_train_dataset,
    validation_data=tf_val_dataset,
    epochs=NUM_EPOCHS,)
```
**Why use this command?**
 Starts model training process
*   **Motive**: Trains the model using the given train and validation dataset, for a predefined amount of epochs
*   **Logic**: The `model.fit` function uses data from `tf_train_dataset` to learn patterns, and after each epoch, computes `loss` and `val_loss` to evaluate performance on the validation split `tf_val_dataset`.

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model_loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
```
**Why use this code?**
Visualizes the training and validation loss values of the models through epochs
*   **Motive**: Allows monitoring the loss during training.
*  **Logic**: This will create a plot of the losses through the training to show the learning trends for both training and validation datasets.

## Evaluation

```python
metric=evaluate.load("seqeval")
```
**Why use this command?**
 Load the evaluation metric to be used to evaluate the model's performance.
*   **Motive**: The model has already been trained, now a metric is required to measure the success of its learning.
*   **Logic**: Loads the sequence evaluation metric called `seqeval` from the `evaluate` library.

```python
ind_to_label={0:'O', 1:'B-PER',2:'I-PER',3:'B-ORG',4:'I-ORG',5:'B-LOC',6:'I-LOC',7:'B-MISC',8:'I-MISC'}
all_predictions = []
all_labels = []
```
**Why use this code?**
 Initializes needed data structures.
*   **Motive**: Initializes a dictionary for mapping indexes to labels and lists for prediction data aggregation.
*  **Logic**:
 * `ind_to_label`: Creates a lookup table to understand each label type during the evaluation.
    * `all_predictions`: Collects predictions after all batches.
    * `all_labels`: Collects correct labels from all validation batches.

```python
for batch in tf_val_dataset:
  logits = model.predict(batch)["logits"]
  labels = batch["labels"].numpy()
  predictions = tf.argmax(logits, axis=-1).numpy()
  for prediction, label in zip(predictions, labels):
    for predicted_idx, label_idx in zip(prediction, label):
      if label_idx == -100:
          continue
      all_predictions.append(ind_to_label[predicted_idx])
      all_labels.append(ind_to_label[label_idx])
```
**Why use this loop?**
 Generates the predictions and aligns them with labels
*   **Motive**: Collects all model predictions and true labels for evaluation by iterating through all the validation data and adding them to lists, not considering the -100 pad value.
*  **Logic**:
    * The loop iterates through every batch from validation data split and generates output `logits` for all of them.
    * By calling `tf.argmax`, we get the indices of maximum probability of `logits`, then, using the defined mapping, the indices get converted to actual names of labels.
   * For each prediction/label pair, padding is excluded and data added to `all_predictions` and `all_labels`.

```python
print(all_predictions)
print(all_labels)
```
**Why use this command?**
 Print out the prediction and the true labels
*  **Motive:** Checks the prediction of the validation dataset against their true labels.
*  **Logic**: Displays the sequence of predicted and the actual labels extracted from all validation batches of the dataset, showcasing the model output.
```python
metric.compute(predictions=[all_predictions], references=[all_labels])
```
**Why use this command?**
 Evaluates overall performance using the defined metric
* **Motive:** Using the metric, get performance statistics for model.
*   **Logic**: This calls the `compute` function to perform a measurement of overall accuracy based on all predictions compared to the true labels from all batches from validation data split, as well as providing statistics per each label.
## Testing

```python
inputs=tokenizer(["Wake Up JoeMarshal, you just got a call from UNESCO for a trip to India"], padding=True,return_tensors="tf")
```
**Why use this command?**
 Preprocess and tokenize test input text, to apply the trained model.
*   **Motive**: Transform the user input text so that it can be fitted in the trained model for inference and prediction.
*  **Logic**: Converts user-defined example text into an input format required by the model through the tokenizer, specifically adds padding in order to be used in batch format.

```python
print(inputs.tokens())
print(inputs.word_ids())
print(inputs['input_ids'])
```
**Why use this command?**
  Visualize input to model
*   **Motive**: Verifies that data has been transformed correctly through tokenizer.
*   **Logic**: Prints out the input sequence in the form of tokens, word ids and numeric ids for checking the transformation.

```python
logits = model(**inputs).logits
print(logits.shape)
print(tf.argmax(logits,axis=-1))
```
**Why use this code?**
 Generate predictions for our test data
*  **Motive:** Use our model in an example scenario
*  **Logic**:
    * `model(**inputs).logits`: Generates logits using the model from the inputs
    *   `logits.shape`: Show how many categories our model predicts.
  *`tf.argmax(logits,axis=-1)` Shows the most probable labels the model predicted.
```python
ind_to_label={0:'O', 1:'B-PER',2:'I-PER',3:'B-ORG',4:'I-ORG',5:'B-LOC',6:'I-LOC',7:'B-MISC',8:'I-MISC'}
out_str=""
current_index=0
for i in range(1,len(inputs.tokens())-1):
  if tf.argmax(logits,axis=-1)[0][i]!=0:
    out_str+=" "+str(inputs.tokens()[i])+"--->"+str(ind_to_label[tf.argmax(logits,axis=-1).numpy()[0][i]])
  else:
    out_str+=" "+str(inputs.tokens()[i])
print(out_str.replace("Ġ",""))
```
**Why use this code?**
 Translates predictions back to labels for user.
*   **Motive**: Converts the numeric output predictions to the corresponding labels.
*  **Logic**:
    * Defines the lookup label table
    *   Iterates through tokens and `argmax` indices of model predictions to transform them into string based representations and then to human readable format.
* `.replace("Ġ","")` removes special chars that were appended due to use of a sub-word tokenizer

## Conclusion
This README thoroughly explained each part of the code, emphasizing why different functions, methods, and commands were used. This documentation should aid in understanding the NER implementation and how different parts of the code work together in processing the data, modeling, training and evaluating our neural network.
```
