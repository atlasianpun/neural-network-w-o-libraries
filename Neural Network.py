# -*- coding: utf-8 -*-


#the data is stored on the drive and hence I mount my drive
# we chose this dataset cause we are alcoholics
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import math
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, OneHotEncoder
import plotly
import plotly.express as px
plotly.offline.init_notebook_mode (connected = True)
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import hyperopt
from hyperopt import hp, fmin, tpe, Trials, space_eval, STATUS_OK

file_path = '/content/drive/MyDrive/ML/Iris.csv' # punya

df = pd.read_csv(file_path)

df = df.sample(frac=1).reset_index(drop=True) # Shuffle

df

class preprocessing_data(object):
    def preprocessing_df(self, df):
        X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
        X = np.array(X)
        one_hot_encoder = OneHotEncoder(sparse=False)
        Y = df.Species
        Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))
        return X,Y

    def split(self,X,Y):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test

class NN(object):
    """A Multilayer Perceptron class.
    """

    def __init__(self, num_inputs=11, hidden_layers=[6, 3], num_outputs=1, activation_fn='sigmoid'):
        """Constructor for the MLP. Takes the number of inputs,
            a variable number of hidden layers, and number of outputs

        Args:
            num_inputs (int): Number of inputs
            hidden_layers (list): A list of ints for the hidden layers
            num_outputs (int): Number of outputs
        """

        self.num_inputs = num_inputs
        self.hidden_layers = hidden_layers
        self.num_outputs = num_outputs
        self.activation_fn = activation_fn

        # create a generic representation of the layers
        layers = [num_inputs] + hidden_layers + [num_outputs]
        # print(f'Length of layers : {len(layers)}' )
        # print(f'Layer format : {layers}')

        # create random weights and biases
        weights = []
        biases = []
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i + 1])  # random weights
            b = np.zeros(layers[i + 1])  # zero biases
            weights.append(w)
            biases.append(b)
        self.weights = weights
        self.biases = biases
        self.m_dw = [np.zeros_like(weights) for weights in self.weights]
        self.v_dw = [np.zeros_like(weights) for weights in self.weights]
        self.m_db = [np.zeros_like(biases) for biases in self.biases]
        self.v_db = [np.zeros_like(biases) for biases in self.biases]
        self.rms_error_test= 0


        # save derivatives per layer - not biases rn
        derivatives = []
        for i in range(len(layers) - 1):
            d = np.zeros((layers[i], layers[i + 1]))
            derivatives.append(d)
        self.derivatives = derivatives

        # save activations per layer
        activations = []
        for i in range(len(layers)):
            a = np.zeros(layers[i])
            activations.append(a)
        self.activations = activations

    def forward_propagate(self, inputs):
        # the input layer activation is just the input itself
        activations_1d = inputs

        # save the activations for backpropagation
        self.activations[0] = activations_1d

        # select activation function
        if self.activation_fn == 'sigmoid':
            activation_func = self._sigmoid
        elif self.activation_fn == 'relu':
            activation_func = self._relu
        elif self.activation_fn == 'tanh':
            activation_func = self._tanh
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

        # iterate through the network layers
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations_1d, w)
            # print(f'Net inputs before bias : {net_inputs}')
            net_inputs = np.add(net_inputs, b)
            # print(f'Net inputs after bias : {net_inputs}')
            # apply selected activation function
            activations_1d = activation_func(net_inputs)
            # print(f'After activation fcn applied : {activations_1d}')
            # save the activations for backpropagation
            self.activations[i + 1] = activations_1d

        # return output layer activation
        return activations_1d

    def back_propagate(self, error):
        """Backpropogates an error signal.
        Args:
            error (ndarray): The error to backprop.
        Returns:
            error (ndarray): The final error of the input
        """

        # iterate backwards through the network layers
        for i in reversed(range(len(self.derivatives))):
            # print(f'-----------{len(self.derivatives)}th layer ----------')
            # get activation for previous layer
            activations = self.activations[i+1]
            if self.activation_fn == "sigmoid":
            # apply sigmoid derivative function
                delta = error * self._sigmoid_derivative(activations)
            elif self.activation_fn == "tanh":
            # apply sigmoid derivative function
                delta = error * self._tanh_derivative(activations)
            elif self.activation_fn == "relu":
            # apply sigmoid derivative function
                delta = error * self._relu_derivative(activations)

            # reshape delta as to have it as a 2d array
            delta_re = delta.reshape(delta.shape[0], -1).T
            # print(f'Deltas : {delta_re}')

            # get activations for current layer
            current_activations = self.activations[i]
            # reshape activations as to have them as a 2d column matrix
            current_activations = current_activations.reshape(current_activations.shape[0],-1)
            # print(f'Current Activations : {current_activations}')

            # save derivative after applying matrix multiplication
            self.derivatives[i] = np.dot(current_activations, delta_re)
            # print(f'Errors : {self.derivatives}')
            # backpropogate the next error
            error = np.dot(delta, self.weights[i].T)
            # print(f'Errors : {error}')



    def fit(self, X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # Initialize Adam optimizer
        print(f"learning rate is {learning_rate}, epochs are {epochs}, hidden layers are {self.hidden_layers}, activation function is {self.activation_fn}")
        # now enter the training loop
        for i in range(epochs):
            accuracy_train = 0
            accuracy_val = 0
            accuracy_test = 0

            # iterate through all the training data
            for x_row, y_row in zip(X_train, Y_train):
                input = x_row
                target = y_row

                # activate the network!
                output = self.forward_propagate(input)

                error = target - output

                self.back_propagate(error)

                # now perform gradient descent on the derivatives
                # (this will update the weights)
                # self.gradient_descent(learning_rate)
                self.gradient_descent()
                # keep track of the MSE for reporting later
                accuracy_train += self._accuracy(target, output)

        for x_row, y_row in zip(X_val, Y_val):
            input = x_row
            target = y_row
            output = self.forward_propagate(input)
            accuracy_val += self._accuracy(target, output)

        # Calculate test error
        for x_row, y_row in zip(X_test, Y_test):
            input = x_row
            target = y_row
            output = self.forward_propagate(input)
            accuracy_test += self._accuracy(target, output)

            # Epoch complete, report the training and test error
            print("Epoch {}: Training accuracy: {}, Test accuracy: {}".format(i + 1, accuracy_train / len(X_train), accuracy_test / len(X_test)))
        print("Training complete!")
        print("=====")
        return (accuracy_val / len(X_train))

    def adam_optimizer(self,learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8 ):
        for i in range(len(self.weights)):
            # Initialize the first moments of gradients for weights and biases
            self.m_dw[i], self.v_dw[i] = np.zeros_like(self.weights[i]), np.zeros_like(self.weights[i])
            self.m_db[i], self.v_db[i] = np.zeros_like(self.biases[i]), np.zeros_like(self.biases[i])

            ## dw, db are from current minibatch
            ## momentum beta 1
            # Calculate the first moment of the gradient for weights using exponential decay
            self.m_dw[i] = beta1*self.m_dw[i] + (1-beta1)*self.derivatives[i]
            # Calculate the first moment of the gradient for biases using exponential decay
            self.m_db[i] = beta1*self.m_db[i] + (1-beta1)*np.sum(self.derivatives[i], axis=0)

            ## rms beta 2
            # Calculate the second moment of the gradient for weights using exponential decay
            self.v_dw[i] = beta2*self.v_dw[i] + (1-beta2)*(self.derivatives[i]**2)
            # Calculate the second moment of the gradient for biases using exponential decay
            self.v_db[i] = beta2*self.v_db[i] + (1-beta2)*(np.sum(self.derivatives[i], axis=0)**2)

            ## bias correction
            # Correct the bias of the first moment estimates for weights and biases
            m_dw_corr = self.m_dw[i]/(1-beta1**(i+1))
            m_db_corr = self.m_db[i]/(1-beta1**(i+1))
            # Correct the bias of the second moment estimates for weights and biases
            v_dw_corr = self.v_dw[i]/(1-beta2**(i+1))
            v_db_corr = self.v_db[i]/(1-beta2**(i+1))

            ## update weights and biases
            # Update the weights using the corrected first and second moment estimates
            self.weights[i] = self.weights[i] - learning_rate*(m_dw_corr/(np.sqrt(v_dw_corr)+epsilon))
            # Update the biases using the corrected first and second moment estimates
            self.biases[i] = self.biases[i] - learning_rate*(m_db_corr/(np.sqrt(v_db_corr)+epsilon))


    def gradient_descent(self, learning_rate=1):

        # update the weights and biases by stepping down the gradient
        for i in range(len(self.weights)):
            self.weights[i] += self.derivatives[i] * learning_rate
            self.biases[i] += self.derivatives[i].sum(axis=0) * learning_rate

    def predict(self, inputs):
        # the input layer activation is just the input itself
        activations_1d = inputs

        # save the activations for backpropagation
        self.activations[0] = activations_1d

        # select activation function
        if self.activation_fn == 'sigmoid':
            activation_func = self._sigmoid
        elif self.activation_fn == 'relu':
            activation_func = self._relu
        elif self.activation_fn == 'tanh':
            activation_func = self._tanh
        else:
            raise ValueError(f"Unknown activation function: {self.activation_fn}")

        # iterate through the network layers
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # calculate matrix multiplication between previous activation and weight matrix
            net_inputs = np.dot(activations_1d, w)
            # print(f'Net inputs before bias : {net_inputs}')
            net_inputs = np.add(net_inputs, b)
            # print(f'Net inputs after bias : {net_inputs}')
            # apply selected activation function
            activations_1d = activation_func(net_inputs)
            # print(f'After activation fcn applied : {activations_1d}')
            # save the activations for backpropagation
            self.activations[i + 1] = activations_1d
        output = np.where(activations_1d >= 0.5, 1, 0)

        # return output layer activation
        return output



    # sigmoid function
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # sigmoid derivative function - - the input is not x but sigmaoid(x)f
    def _sigmoid_derivative(self,x):
        return x * (1-x)


    # Tanh function
    def _tanh(self, x):
        return np.tanh(x)

    #tanh derivative function - the input is not x but tanh(x)
    def _tanh_derivative(self, x):
        return (1-x**2)


    # relu activation function
    def _relu(self, x):
        return np.maximum(0, x)

    # relu derivative function - the input is not x but relu(x)
    def _relu_derivative(self,x):
        return np.where(x > 0, 1, 0)

    #RMSE
    def _mse(self, target, output):
        return np.average((target - output) ** 2)

    def _accuracy(self, target, output):
        binary_output = np.where(output >= 0.5, 1, 0)

        num_matches = np.sum(binary_output == target)

        # Calculate the percentage of matching elements
        percentage = (num_matches / len(target))

        return percentage

preprocess = preprocessing_data()
X, Y = preprocess.preprocessing_df(df)
X_train, X_val, X_test, Y_train, Y_val, Y_test = preprocess.split(X,Y)

#defining the necessary hyperopt obhective function
def objective_fun(params):
    #defining the initial version of the neural network
    nn = NN(num_inputs=4, hidden_layers=[params['l1'], params['l2']], num_outputs=3, activation_fn=params['activation_fn'])
    accuracy = nn.fit(X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs = 40, learning_rate=params['learning_rate'], beta1=0.9, beta2=0.999, epsilon=1e-8)

    return {'loss': 1 - accuracy,
          'status': STATUS_OK,
          'model': nn,
          'params': params}

#defining paramter space for the hyperot lib to work out of
param_space = {
      "activation_fn": hp.choice("activation",['relu', 'sigmoid', 'tanh']),
      "learning_rate": hp.uniform("learning_rate",0.001,1),
      "l1": hp.choice("l1", range(10,100)),
      "l2": hp.choice("l2", range(10,100)),
  }

trials = Trials()

best_params = fmin(
  fn=objective_fun,
  space=param_space,
  algo=tpe.suggest,
  max_evals=30,
  trials=trials)

#obtain the best paramteter
print(best_params)

#testing
nn  = NN(num_inputs=4, hidden_layers=[20, 40], num_outputs=3, activation_fn='sigmoid')
accuracy_error = nn.fit(X_train, Y_train, X_val, Y_val, X_test, Y_test, epochs = 40, learning_rate = 0.914712626789224, beta1=0.9, beta2=0.999, epsilon=1e-8)
Ycap_test = nn.predict(X_test)

#dispaly the output
print(Ycap_test)

print(Y_test)