
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from math import exp

import sklearn
from sklearn.metrics import accuracy_score

import random




class Artificial_Neural_Network:
  """
  This class is a personalized MLP, it has one neuron per output and the
  class prediction always depends on the calculation of the sigmoidal activation function

  This MLP class is not optimized, it has a considerably long execution time, keep in
  mind that this code is for educational purposes, not productive
  """

  AJ = None
  AK = None
  HJ = None


  def __init__(self, 
               input_size:int, 
               hidden_size:int, 
               output_size:int) -> None:
    
    if not isinstance(input_size, int):
      raise TypeError(f"An Integer is expected for 'input_size', got a {type(input_size)}")
    if not isinstance(hidden_size, int):
      raise TypeError(f"An Integer is expected for 'hidden_size', got a {type(hidden_size)}")
    if not isinstance(output_size, int):
      raise TypeError(f"An Integer is expected for 'output_size', got a {type(output_size)}")
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
  
    if (self.__class__.AJ is None) and (self.__class__.AK is None) and (self.__class__.HJ is None):
      self.__class__.AJ = np.zeros((self.hidden_size,1)) 
      self.__class__.AK = np.zeros((self.output_size,1))
      self.__class__.HJ = np.zeros((self.hidden_size+1,1))


  ####### Instance Method #######
  def graph_neural_network(self) -> None:
    """
    This function makes a graph with the neural network topology
      input layer -> hidden layer -> output layer
    """

    total_height = self.input_size + max(self.hidden_size, self.output_size) #Calculate the total height of the graphic
    center_y = total_height / 2 #Calculate the vertical position of the center of the graphic
    fig = plt.figure(figsize=(10, 5))

    ##Add the nodes of the input layer
    for i in range(self.input_size):
        plt.scatter(0, i+0.5+(center_y-self.input_size/2), s=500, c='black')
        
    ##Add the nodes of the hidden layer
    for i in range(self.hidden_size):
        plt.scatter(2, i+0.5+(center_y-self.hidden_size/2), s=500, c='black')
        
        ##Connect the nodes
        for j in range(self.input_size):
            plt.plot([0, 2], [j+0.5+(center_y-self.input_size/2), i+0.5+(center_y-self.hidden_size/2)], c='black')
            
    ##Add the nodes of the output layer
    for i in range(self.output_size):
        plt.scatter(4, i+0.5+(center_y-self.output_size/2), s=500, c='black')
        
        ##Connect the nodes
        for j in range(self.hidden_size):
            plt.plot([2, 4], [j+0.5+(center_y-self.hidden_size/2), i+0.5+(center_y-self.output_size/2)], c='black')

    plt.axis('off')
    plt.show()
    
    
  ####### Instance Method #######
  def fit(self,
          features:np.ndarray, 
          labels:np.ndarray,
          alpha:float, 
          iterations:int, 
          MSE_stop:float) -> np.ndarray and np.ndarray and np.ndarray:
    """
    This function is in charge to perform the training of the neural network
      Parameters:
        features -> array with the dataset features
        labels -> array with the dataset class labels
        alpha -> learning rate
        iterations -> training repetitions
        MSE_stop -> target loss to stop training
      Returns:
        YK_T -> training predictions
        w -> hidden layer weights
        c -> output layer weights
    """

    if not isinstance(features, np.ndarray):
      raise TypeError(f"Features should be added as a 'np.ndarray', instead got a {type(features)}")
    if not isinstance(labels, np.ndarray):
      raise TypeError(f"Labels should be added as a 'np.ndarray', instead got a {type(labels)}")
    
    rows = features.shape[0]
    inputs = self.input_size
    hidden = self.hidden_size
    outputs = self.output_size

    ##Aggregations
    AJ = self.__class__.AJ
    AK = self.__class__.AK 

    ##Activations
    HJ = self.__class__.HJ 
    ####HJ[hidden] = 1 #Add 'BIAS'
    YK_T = np.zeros((rows,outputs))

    a = alpha #Definition of the Learning Rate

    ##Neural Weights
    w = np.random.rand(hidden,inputs)
    c = np.random.rand(outputs, hidden+1)

    YD_T = labels #Real Value
    YD_T = YD_T.reshape((YD_T.shape[0], 1))

    ##Losses
    E = np.zeros((rows,outputs))
    MSE = np.zeros((outputs,1))
    MSE_T = np.zeros((outputs,1))

    max_iter = iterations

    for iter in range(max_iter):

      if (iter+1 == max_iter):
        print("\nITERATIONS LIMIT REACHED\n")
        break

      MSE[:] = 0
      MSE_SUM = 0
      
      for i in range(0, len(features)):
        ##Hidden Layer Agregation
        AJ[:] = 0
        for k in range(hidden):
          for j in range(0, len(w[0])):
            AJ[k] = AJ[k]+features[i][j]*w[k][j]
        ##Hidden Layer Activation Function
        for j in range(hidden):
          HJ[j] = 1/(1+exp(-AJ[j]))
        ##Output Layer Agregation
        AK[:] = 0
        for k in range(outputs):
          for j in range(0, len(c[0])):
            AK[k] = AK[k]+HJ[j]*c[k][j]
        ##Output Layer Activation Function
        for k in range(outputs):
          YK_T[i][k] = 1/(1+exp(-AK[k])) 
        
        ##Loss
        for k in range(outputs):
          E[i][k] = YD_T[i][k]-YK_T[i][k]
          
        ##Output Layer Training
        for k in range(outputs):
          for j in range(len(c[0])):
            ####c[k][j] = c[k][j]+alfa*E[i][k]*1*HJ[j]
            c[k][j] = c[k][j]+a*E[i][k]*(YK_T[i][k]*(1-YK_T[i][k]))*HJ[j]
        ##Hidden Layer Training
        for e in range(outputs):
          for k in range(hidden):
            for j in range(len(w[0])):
              ####w[k][j] = w[k][j]+alfa*E[i][e]*1*c[e][k]*1*X[i][j]
              w[k][j] = w[k][j]+a*E[i][e]*(YK_T[i][e]*(1-YK_T[i][e]))*c[e][k]*(HJ[k]*(1-HJ[k]))*features[i][j]
      
      ##MSE
      for k in range(outputs):
        for i in range(len(E)):
          MSE[k] = MSE[k]+((E[i][k]**2/2))

      MSE_T = np.hstack((MSE_T, MSE))
      MSE_SUM = np.sum(MSE)

      if MSE_SUM <= MSE_stop:
        break

    ##MSE Graphic
    print("MSE =", MSE_SUM)
    plt.figure(figsize=(10,5))
    plt.plot(MSE_T[0], 'k')
    plt.title('Loss (MSE) During Training')


    return YK_T, w, c


  ####### Instance Method #######
  def predict(self,
              features:np.ndarray,
              w:np.ndarray, 
              c:np.ndarray) -> np.ndarray:
    """
    This functions performs the prediction of the trained MLP model
      Parameters:
        features -> array with the dataset features
        w -> hidden layer wieghts
        c -> output layer weights
      Returns:
        YK_V -> array with the predicted labels
    """

    if not isinstance(features, np.ndarray):
      raise TypeError(f"Features should be added as a 'np.ndarray', instead got a {type(features)}")
    if not isinstance(w, np.ndarray):
      raise TypeError(f"'W' must be a matrix <class 'np.ndarray'>, instead got a {type(w)}")
    if not isinstance(c, np.ndarray):
      raise TypeError(f"'C' must be a matrix <class 'np.ndarray'>, instead got a {type(c)}")
    
    rows = features.shape[0]
    outputs = self.output_size
    hidden = self.hidden_size

    ##Aggregations
    AJ = self.__class__.AJ  
    AK = self.__class__.AK  

    ##Activations
    HJ = self.__class__.HJ 
    ####HJ[hidden] = 1 #Add 'BIAS'
    YK_V = np.zeros((rows,outputs))

    for i in range(0, features.shape[0]):
        ##Hidden Layer Agregation
        AJ[:] = 0
        for k in range(hidden):
            for j in range(0, len(w[0])):
                AJ[k] = AJ[k]+features[i][j]*w[k][j]
        ##Hidden Layer Activation Function
        for j in range(hidden):
            HJ[j] = 1/(1+np.exp(-AJ[j]))
        ##Output Layer Agregation
        AK[:] = 0
        for k in range(outputs):
            for j in range(0, len(c[0])):
                AK[k] = AK[k]+HJ[j]*c[k][j]
        ##Output Layer Activation Function
        for k in range(outputs):
            YK_V[i][k] = 1/(1+exp(-AK[k])) 


    return YK_V


  @staticmethod
  def accuracy(true_values:np.ndarray, 
               predicted_values:np.ndarray, 
               scaler:sklearn.preprocessing._data.MinMaxScaler) -> float:
    """
    This function calculates the general accuracy of the MLP model
      Parameters:
        true_values -> aray with the real values
        predicted_values -> array with the model predictions
        scaler -> Scaler object that was used in the data preprocessing
      Returns:
        accuracy_score -> general accuracy
    """

    if not isinstance(true_values, np.ndarray):
      raise TypeError(f"Features should be added as a 'np.ndarray', instead got a {type(true_values)}")
    if not isinstance(predicted_values, np.ndarray):
      raise TypeError(f"Predictions should be added as a 'np.ndarray', instead got a {type(predicted_values)}")
    
    n_partitions = len(np.unique(true_values)) #Define the number of partitions
    bins = np.linspace(0, 1, n_partitions + 1) #Create ranges' limits
    arr_class = np.digitize(predicted_values, bins).astype(float) #Classify the data in the ranges
    arr_class = (arr_class - 1) / (n_partitions - 1) #Asign the corresponding class


    return accuracy_score(scaler.inverse_transform(true_values.reshape((true_values.shape[0], 1))), 
                          scaler.inverse_transform(arr_class))


  ####### Instance Method #######
  def graph_results(self,
                    true_values:np.ndarray, 
                    predicted_values:np.ndarray,
                    title:str) -> None:
    """
    This function makes a graph with the prediction results of the MLP model
      Parameters:
        true_values -> array with the real values
        predicted_values -> array with the model predictions
    """
    
    outputs = self.output_size

    n_partitions = len(np.unique(true_values)) #Define the number of partitions
    bins = np.linspace(0, 1, n_partitions + 1) #Create ranges' limits

    ##Real VS Prediction
    aux_YD = true_values
    aux_YK = np.zeros((outputs, len(true_values)))
    for i in range(outputs):
      for j in range(len(true_values)):
        aux_YK[i][j] = predicted_values[j][i]   

    plt.figure(figsize=(10,5))
    for i in range(len(bins)):
      plt.plot(np.array(range(len(aux_YD))),np.repeat(bins[i], len(aux_YD)), 'k')
    plt.plot(aux_YD, '*b', label='Real Value')
    plt.plot(aux_YK[0], '*r', label='Model Prediction')
    plt.legend(loc='best')
    plt.title(title)


  @staticmethod
  def probas(true_values:np.ndarray, 
             predicted_values:np.ndarray, 
             names:np.ndarray) -> pd.core.frame.DataFrame:
    """
    This function gives the probability of a predictions really
    belonging to the predicted class
      Parameters:
        true_values -> array with the real values
        predicted_values -> array with the model predictions
      Returns:
        table -> a table with the real class, predicted class, and probability
    """

    if not isinstance(true_values, np.ndarray):
      raise TypeError(f"Features must be added as a 'np.ndarray', instead got a {type(true_values)}")
    if not isinstance(predicted_values, np.ndarray):
      raise TypeError(f"Predictions must be added as a 'np.ndarray', instead got a {type(predicted_values)}")
    if not isinstance(names, np.ndarray):
      raise TypeError(f"Target Names must be added as a 'np.ndarray', instead got a {type(names)}")
    
    n_partitions = len(np.unique(true_values)) #Define the number of partitions
    bins = np.linspace(0, 1, n_partitions + 1) #Create ranges' limits
    arr_class = np.digitize(predicted_values, bins).astype(float) #Classify the data in the ranges
    arr_class = (arr_class - 1) / (n_partitions - 1) #Asign the corresponding class

    real = true_values.reshape((true_values.shape[0], 1))
    clasif = arr_class.reshape((arr_class.shape[0], 1))
    probas = 1-np.abs(clasif-predicted_values)
    table = pd.DataFrame(np.hstack((clasif, probas, real)), 
                        columns=['Model Classification', 'Probability', 'Real Classification'])

    valores_unicos = table['Real Classification'].unique()
    head_names = names
    mapeo = {valor: nombre for valor, nombre in zip(valores_unicos, head_names)}
    table['Model Classification'] = table['Model Classification'].replace(mapeo)
    table['Real Classification'] = table['Real Classification'].replace(mapeo)


    return table
