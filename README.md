# Object Classification using AlexNet and Asirra cats vs dogs

## Collaborator
**Hyunjae Gil**: neural net construction, train parameter setting \
**Chan Lee**: dataset loading, neural net tuning, training on server, documentation

## Introduction
Object Classification code with neural network model of AlexNet and Asirra cats vs dogs as dataset
> ImageNet dataset has way far large file (need more than 300GB), so that Asirra dataset was used 

![11-Figure14-1](https://user-images.githubusercontent.com/59006698/88611894-00df7280-d0c5-11ea-8a94-722ed673d18b.png)
![image-asset](https://user-images.githubusercontent.com/59006698/88609630-937d1300-d0bf-11ea-8c95-bb7a9d0c6e09.png)

## Brief Explanation
> assira.py for data loading, asirra_cnn.py for training, neural net construction in function_used.py
### Load Data: assira.py
Load the Asirra Dogs vs. Cats data subset from disk and perform preprocessing for training AlexNet.\
Such as resizing, random crop for data augmentation, labeling 
### Construct Neural Net: function_used.py
> keras has used for implementing neural network

Five convolution layers and three fully connected layers, Three pooling layers, two normalization layers and one dropout layer 
### Train & Val : asirra_cnn.py
1. Load augmented dataset 
2. Set training hyper parameters
```python
    learning_rate = 0.001
    training_epoch = 5
    batch_size = 64
    display_step = 20
```
3. Build Model, initialize a session and start training
4. Save the validation result with png file
