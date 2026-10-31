# Deep-Neural-Networks-CW
Coursework project for INM702: Mathematics and Programming for AI at City, University of London. The project consists of 2 tasks: the first one is about building a neural network from scratch and the second one is about implementing deep neural networks using PyTorch.

Dataset: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Extended Dataset: https://www.kaggle.com/datasets/alfathterry/telco-customer-churn-11-1-3


# Intructions for rerunning:

# Task 1:

# Imports and Requirement

1) kagglehub: Version 0.3.13
2) numpy: Version 2.3.5
3) pandas: Version 2.3.3
4) sklearn: Version 1.8
5) matplotlib: Version 3.10.8

# Running the code

1) Open Jupyter notebook, and navigate to task 1/T1_1-0-0_Custom-Neural-Network.ipynb and open it
2) Change the Random State or seed value and then reset kernal and run all for the results

Note: No other external dataset imports are required as we are directly importing from kaggle using kagglehub

# Classes and their methods and parameters


## ReLU

Methods:
1) forward_pass: Calculates output of ReLU when doing forward propagation
2) backward_pass: Calculates Derivates of ReLU when doing backpropagation


## Sigmoid

Methods:
1) forward_pass: Calculates output of Sigmoid when doing forward propagation
2) backward_pass: Calculates Derivates of Sigmoid when doing backpropagation


## Softmax Layer

Methods:
1) forward_pass: Calculates output of Softmax when doing forward propagation
2) backward_pass: Calculates Derivates of Softmax when doing backpropagation


## Optimizers

Params:
1) lr: Learning rate (Default = 0.01)
2) mu: Momentum Coefficient (Default = 0.9) (Exclusive for SGDMomentum())

Methods:
1) update_parameters: Updates the weights and biases


## Dropout

Params:
1) prob: probability of disabling the neuron (Default = 0.5)
2) inv: Applying inverted dropout (Default = True)

Methods:
1) forward_pass: Calculates the mask for dropout and apply it when doing forward propagation
2) backward_pass: Apply the mask obtained during forward_pass onto the derivate when doing backpropagation



## Modification of the Linear Model

Params:
1) input: The number of input nodes
2) output: The number of output nodes
3) act: The activation function (None, ReLU(), Sigmoid()) (Default = None)
4) drop: The dropout function (None, Dropout()) (Default = None)
5) lambda_l1: L1 value (Default = 0.0)
6) lambda_l2: L2 value (Default = 0.0)
7) uni: Unique Value to identify the Linear Layer (Default = 0)

Methods:
1) forward_pass: Calculates the output by applying forward propagation of Activation Function and Dropout in order when doing forward propagation
2) backward_pass: Calculates the derivate of input by applying backpropagation of Dropout and Activation Function in order when doing backpropagation


## Custom Fully Parameterized Neural Network Class

Params:
1) inputLayers: An array of Layers, aka architecture of the neural network
2) loss: Loss Function (BinaryCrossEntropyLoss(), CategoricalCELoss()) (Default = BinaryCrossEntropyLoss())
3) optimize: Optimizer function (None,SGDOptimizer(),SGDMomentum()) (Default = None)

Methods:
1) forward_pass: Iterates through the inputLayer and applies forward pass for each layer
2) backward_pass: Iterates backwards throught the inputLayer and applies back pass for each layer
3) singleTrainStep: Trains the model and calculates the predicted Y value, updates parameters and Calcuates Loss and Accuracy
4) train: Runs singleTrainStep for specified number of epochs (Default = 100), prints loss and accuracy graph and prints the highest accuracy achieved
5) test: Returns Accuracy and loss for the trained model by running through forward pass

# Using the Neural Network

## Example Code:

```
X = X_train
Y = y_train

model4 = CustomNeuralNetwork(
            inputLayers = [
                CustomLinearImplementation(30,60,act=ReLU(),drop=Dropout(0.25,inv=False),uni=1),
                CustomLinearImplementation(60,80,act=Sigmoid(),drop=Dropout(0.25,inv=False),uni=2),
                CustomLinearImplementation(80,40,act=ReLU(),drop=Dropout(0.25,inv=False),uni=3),
                CustomLinearImplementation(40,20,act=Sigmoid(),drop=Dropout(0.25,inv=False),uni=4),
                CustomLinearImplementation(20,10,act=ReLU(),drop=Dropout(0.25,inv=False),uni=5),
                CustomLinearImplementation(10,1,act=Sigmoid(),uni=6)
                ],
            loss=BinaryCrossEntropyLoss(),
            optimize=SGDOptimizer(lr=0.01))

model4.train(X,Y,epochs=100)

model4.test(X_test,y_test)
model4.test(X_val,y_val)
```

## Instructions

1) the inputLayers needs to be an array of CustomLinearImplementation, but can also have an individual activation for output layer (but ensure the output layer has the appropriate number of output nodes)
2) Each CustomLinearImplemtation layer requires the number of input and output nodes, by default the activation and dropout are None.
3) Adjust the loss depending on if the classification is binary or multi class by using BinaryCrossEntropyLoss() and CategoricalCELoss() respectively
4) The Optimizer is None by default, but we have implemented SGDOptimizer() and SGDMomentum() classes to use for this. (Furthermore, you can change the learning rate here.
5) When training, you can specify the number of epochs the model should train for.
6) When testing, input X test dataset and input Y test dataset

Note: Please ensure that the number of nodes for input layer is the same as the shape of X (features)


# Task 2

# Requirements and imports

1) torch: Version 2.9.1
2) kagglehub: Version 0.3.13
3) numpy: Version 2.3.5
4) pandas: Version 2.3.3
5) sklearn: Version 1.8
6) matplotlib: Version 3.10.8

# Running Code

1) Open Jupyter notebook, and navigate to task 2/T2_1-0-0_Baseline-Imp.ipynb and open it. Change the superSeed value (we ran all results using seed 42), restart kernal and run all. This will provide the baseline metrics.
2) Navigate to task 2/T2_1-1-0_Improved-Arch-Imp.ipynb and open it. Change the superSeed value, restart kernal and run all. This will provide the Improved Architecture metrics.
3) Navigate to task 2/T2_1-2-0_CNN-Arch-Imp.ipynb and open it. Change the superSeed value, restart kernal and run all. This will provide the CNN Architecture metrics.
4) Navigate to task 2/T2_1-2-1_CNN-Arch-Imp_Extended-Dataset.ipynb and open it. Change the superSeed value, restart kernal and run all. This will provide the CNN Architecture metrics for the extended dataset.

