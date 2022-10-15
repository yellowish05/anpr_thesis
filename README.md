# anpr_thesis

dataset = data-complete folder not uploaded*


STEP 1: IMPORT NEEDED LIBRARIES

IMPORTANT MODULES




	1. Import numpy as np
	 
		- bring the NumPy library into your current environment.
		- for working with numerical data in Python.
		
	2. Import matplotlib.pyplot as plt
	
		- collection of command style functions that make matplotlib work like MATLAB.
		
	3. Import os
	
		- function for creating and removing a directory (folder), fetching its contents, changing and identifying the current directory, etc.
		
	4. Import cv2 
	
		- OpenCV is tool for image processing and performing computer vision tasks. It is an open-source library that can be used to perform tasks like face detection, objection tracking, landmark detection, and much more.
		
	5. Import random
	
		- Python Random module is an in-built module of Python which is used to generate random numbers.
		 
	6. Import pickle
	
		- Pickle is used for serializing and de-serializing Python object structures, also called marshalling or flattening. 
		- Serialization refers to the process of converting an object in memory to a byte stream that can be stored on disk or sent over a network.
		
	7. from tensorflow.keras.utils import to_categorical
	
		- provides functions to perform actions on numpy arrays.
		- This allows using categorical cross entropy as the cost function
		


	1. Import keras
	
		○ library for developing and evaluating deep learning models. It is part of the TensorFlow library and allows you to define and train neural network models in just a few lines of code.
		
	2. Sequential 
		○ The sequential API allows you to create models layer-by-layer for most problems. 
		○ It is limited in that it does not allow you to create models that share layers or have multiple inputs or outputs.
		
	3. Dense ()
		○ Construct a dense layer with the hidden layers and units
	
	4. Dropout ()
		○ A Simple Way to Prevent Neural Networks from Overfitting.
		
		Overfitting refers to a model that models the training data too well.
		
		Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. 
		
		○ Dropout is easily implemented by randomly selecting nodes to be dropped out with a given probability (e.g., 20%) in each weight update cycle.
		
		○ Dropout is a recently introduced regularization technique where randomly selected activations are set to 0 during the training, so that the model becomes less sensitive to specific weights in the network
		
	5. Flatten ()
		○ flattens the multi-dimensional input tensors into a single dimension, so you can structure your input layer and build your neural network model, then pass those data into every single neuron of the model effectively. 
	
	6. Conv2d ()
		○ Construct a two-dimensional convolutional layer with the number of filters, filter kernel size, padding, and activation function like arguments.
		
	7. max_pooling2d ()
		○ Construct a two-dimensional pooling layer using the max-pooling algorithm.
	
	8. Batch normalization
		○ to normalize the input layer as well as hidden layers by adjusting mean and scaling of the activations. Because of this normalizing effect with additional layer in deep neural networks, the network can use higher learning rate without vanishing or exploding gradients.
		○ Batch normalization is a type of supplemental layer which adaptively normalizes the input values of the following layer, mitigating the risk of overfitting, as well as improving gradient flow through the network, allowing higher learning rates, and reducing the dependence on initialization
		
	9. LeakyReLU
		○ It fixes the “dying ReLU” problem, as it doesn't have zero-slope parts. 
		○ It speeds up training. 
		○ the usage of ReLU helps to prevent the exponential growth in the computation required to operate the neural network.


STEP 2: DEFINE DATA PATH

STEP 3: ACCESING FOLDERS DATASET 

STEP 4: SHOWING TESTING AND TRAINING SHAPE
	- TRAINING: total of 1111 images in 36 classes, sizes: 28x28x1
	- TESTING: total of 673 images in 36 classes, sizes: 28x28x1

STEP 5: SHOWING OUTPUT CLASSES IN DATASET 

STEP 6: SHOWING FIRST IMAGE IN TESTING AND TRAINING DATASET


STEP 7: CONVERT INT TO FLOAT


STEP 8: CONVERT TO ONE HOT ENCODING

	- REASON for converting categorical to one hot encoding: convert the categorical data in one hot encoding is that machine learning algorithms cannot work with categorical data directly.


STEP 9: SPLITTING DATASET

	- For the model to generalize well, you split the training data into two parts, one designed for training and another one for validation.
		•  you will train the model on 80% of the training data and validate it on 20% of the remaining training data.
		• This will also help to reduce overfitting since you will be validating the model on the data it would not have seen in training phase, which will help in boosting the test performance.


NOTE:
	
Available data are typically split into three sets: 
		a. Training
		b. Validation
		c. test set. 
		
	- A training set is used to train a network, where loss values are calculated via forward propagation and learnable parameters are updated via backpropagation. 
	
	- A validation set is used to monitor the model performance during the training process, fine-tune hyperparameters, and perform model selection. 
	
	- A test set is ideally used only once at the very end of the project in order to evaluate the performance of the final model that is fine-tuned and selected on the training process with training and validation sets


STEP 10: TRAINING 

three convolutional layers:
	○ The first layer will have 32-3 x 3 filters,
	○ The second layer will have 64-3 x 3 filters and
	○ The third layer will have 128-3 x 3 filters


STEP 11: SAVE THE MODEL 
car_dropout.h5


STEP 12: MODEL EVALUATION 

STEP 13: PLOTTING


STEP 14: PREDICTING MODELS
correct and incorrect labels


STEP 15: CLASSIFICATION REPORT


