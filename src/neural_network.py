
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from math import exp

import sklearn
from sklearn.metrics import accuracy_score

import random




class RNA:

  AJ = None
  AK = None
  HJ = None


  def __init__(self, 
               input_size:int, 
               hidden_size:int, 
               output_size:int) -> None:
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
  
    if (self.__class__.AJ is None) and (self.__class__.AK is None) and (self.__class__.HJ is None):
      self.__class__.AJ = np.zeros((self.hidden_size,1)) 
      self.__class__.AK = np.zeros((self.output_size,1))
      self.__class__.HJ = np.zeros((self.hidden_size+1,1))


  #######Método de Instancia#######
  def graph_neural_network(self) -> None:

      total_height = self.input_size + max(self.hidden_size, self.output_size) #Calcular la altura total de la gráfica
      center_y = total_height / 2 #Calcular la posición vertical del centro de la gráfica
      fig = plt.figure(figsize=(10, 5)) #Crear una figura y definir el tamaño

      ##Agregar los nodos de la primera capa
      for i in range(self.input_size):
          plt.scatter(0, i+0.5+(center_y-self.input_size/2), s=500, c='black')
          
      ##Agregar los nodos de la segunda capa
      for i in range(self.hidden_size):
          plt.scatter(2, i+0.5+(center_y-self.hidden_size/2), s=500, c='black')
          
          ##Conectar los nodos de la primera y segunda capa
          for j in range(self.input_size):
              plt.plot([0, 2], [j+0.5+(center_y-self.input_size/2), i+0.5+(center_y-self.hidden_size/2)], c='black')
              
      ##Agregar los nodos de la tercera capa
      for i in range(self.output_size):
          plt.scatter(4, i+0.5+(center_y-self.output_size/2), s=500, c='black')
          
          ##Conectar los nodos de la segunda y tercera capa
          for j in range(self.hidden_size):
              plt.plot([2, 4], [j+0.5+(center_y-self.hidden_size/2), i+0.5+(center_y-self.output_size/2)], c='black')

      ##Desactivar los ejes
      plt.axis('off')

      ##Mostrar la figura
      plt.show()
    
    
  #######Método de Instancia#######
  def fit(self,
          features:np.ndarray, 
          labels:np.ndarray,
          hidden:int, 
          alpha:float, 
          iterations:int, 
          ECM_stop:float) -> np.ndarray and np.ndarray and np.ndarray: 

    rows = features.shape[0]
    inputs = self.input_size
    outputs = self.output_size

    ##Agregación Capa Oculta
    AJ = self.__class__.AJ
    AK = self.__class__.AK 

    ##Funciones de Activación de la Salida
    HJ = self.__class__.HJ 
    ####HJ[hidden] = 1 #Agregar 'BIAS'
    
    YK_T = np.zeros((rows,outputs))

    a = alpha #Factor de Aprendizaje

    ##Pesos Neuronales
    w = np.random.rand(hidden,inputs)
    c = np.random.rand(outputs, hidden+1)

    YD_T = labels #Valor Deseado
    YD_T = YD_T.reshape((YD_T.shape[0], 1))

    ##Errores
    E = np.zeros((rows,outputs))
    ECM = np.zeros((outputs,1))
    ECM_T = np.zeros((outputs,1))

    max_iter = iterations

    for iter in range(max_iter):

      if (iter+1 == max_iter):
        print("\nSE LLEGÓ AL LÍMITE DE ITERACIONES\n")
        break

      ECM[:] = 0
      SUMA_ECM = 0
      
      for i in range(0, len(features)):
        ##Agregación de Capa Oculta
        AJ[:] = 0
        for k in range(hidden):
          for j in range(0, len(w[0])):
            AJ[k] = AJ[k] +features[i][j]*w[k][j]
        ##Función de Activación de Capa Oculta
        for j in range(hidden):
          HJ[j] = 1/(1+exp(-AJ[j]))
        ##Agregación de Capa de Salida
        AK[:] = 0
        for k in range(outputs):
          for j in range(0, len(c[0])):
            AK[k] = AK[k]+HJ[j]*c[k][j]
        ##Función de Activación de Capa de Salida
        for k in range(outputs):
          YK_T[i][k] = 1/(1+exp(-AK[k])) 
        
        ##Error
        for k in range(outputs):
          E[i][k] = YD_T[i][k]-YK_T[i][k]
          
        ##Entrenamiento de Capa de Salida
        for k in range(outputs):
          for j in range(len(c[0])):
            ####c[k][j] = c[k][j]+alfa*E[i][k]*1*HJ[j]
            c[k][j] = c[k][j]+a*E[i][k]*(YK_T[i][k]*(1-YK_T[i][k]))*HJ[j]
        ##Entrenamiento de Capa Oculta
        for e in range(outputs):
          for k in range(hidden):
            for j in range(len(w[0])):
              ####w[k][j] = w[k][j]+alfa*E[i][e]*1*c[e][k]*1*X[i][j]
              w[k][j] = w[k][j]+a*E[i][e]*(YK_T[i][e]*(1-YK_T[i][e]))*c[e][k]*(HJ[k]*(1-HJ[k]))*features[i][j]
      
      ##ECM
      for k in range(outputs):
        for i in range(len(E)):
          ECM[k] = ECM[k]+((E[i][k]**2/2))

      ECM_T = np.hstack((ECM_T, ECM))
      SUMA_ECM = np.sum(ECM)

      if SUMA_ECM <= ECM_stop:
        break

    ##Gráfica ECM
    print("ECM = ", SUMA_ECM)
    plt.figure(figsize=(10,5))
    plt.plot(ECM_T[0], 'k')
    plt.title('ECM Entrenamiento')


    return YK_T, w, c


  #######Método de Instancia#######
  def predict(self,
              features:np.ndarray,
              w:np.ndarray, 
              c:np.ndarray) -> np.ndarray:

      filas = features.shape[0]
      salidas = self.output_size
      hidden = self.hidden_size

      ##Agregación Capa Oculta
      AJ = self.__class__.AJ  
      AK = self.__class__.AK  

      ##Funciones de Activación de la Salida
      HJ = self.__class__.HJ 
      ####HJ[hidden] = 1 #Agregar 'BIAS'

      YK_V = np.zeros((filas,salidas))

      for i in range(0, features.shape[0]):
          ##Agregación Capa Oculta
          AJ[:] = 0
          for k in range(hidden):
              for j in range(0, len(w[0])):
                  AJ[k] = AJ[k]+features[i][j]*w[k][j]
          ##Función de Activación de Capa Oculta
          for j in range(hidden):
              HJ[j] = 1/(1+np.exp(-AJ[j]))
          ##Agregación de Capa de Salida
          AK[:] = 0
          for k in range(salidas):
              for j in range(0, len(c[0])):
                  AK[k] = AK[k]+HJ[j]*c[k][j]
          ##Función de Activación de Capa de Salida
          for k in range(salidas):
              YK_V[i][k] = 1/(1+exp(-AK[k])) 


      return YK_V


  @staticmethod
  def accuracy(true_values:np.ndarray, 
               predicted_values:np.ndarray, 
               scaler:sklearn.preprocessing._data.MinMaxScaler) -> float:

      n_partitions = len(np.unique(true_values)) #Definir la cantidad de particiones
      bins = np.linspace(0, 1, n_partitions + 1) #Crear los límites de los rangos
      arr_class = np.digitize(predicted_values, bins).astype(float) #Clasificar los datos en los rangos
      arr_class = (arr_class - 1) / (n_partitions - 1) #Asignar la clase correspondiente


      return accuracy_score(scaler.inverse_transform(true_values.reshape((true_values.shape[0], 1))), 
                            scaler.inverse_transform(arr_class))


  #######Método de Instancia#######
  def graph_results(self,
                    true_values:np.ndarray, 
                    predicted_values:np.ndarray) -> None:
    
    outputs = self.output_size

    n_partitions = len(np.unique(true_values)) #Definir la cantidad de particiones
    bins = np.linspace(0, 1, n_partitions + 1) #Crear los límites de los rangos

    ##Deseado VS Respuesta
    aux_YD = true_values
    aux_YK = np.zeros((outputs, len(true_values)))
    for i in range(outputs):
      for j in range(len(true_values)):
        aux_YK[i][j] = predicted_values[j][i]   

    plt.figure(figsize=(10,5))
    for i in range(len(bins)):
      plt.plot(np.array(range(len(aux_YD))),np.repeat(bins[i], len(aux_YD)), 'k')
    plt.plot(aux_YD, '*b', label='Valor Deseado')
    plt.plot(aux_YK[0], '*r', label='Respuesta de Red')
    plt.legend(loc='best')
    plt.title('Deseado vs Respuesta')


  @staticmethod
  def probas(true_values:np.ndarray, 
             predicted_values:np.ndarray, 
             names:np.ndarray) -> pd.core.frame.DataFrame:

      n_partitions = len(np.unique(true_values)) #Definir la cantidad de particiones
      bins = np.linspace(0, 1, n_partitions + 1) #Crear los límites de los rangos
      arr_class = np.digitize(predicted_values, bins).astype(float) #Clasificar los datos en los rangos
      arr_class = (arr_class - 1) / (n_partitions - 1) #Asignar la clase correspondiente

      real = true_values.reshape((true_values.shape[0], 1))
      clasif = arr_class.reshape((arr_class.shape[0], 1))
      probas = 1-np.abs(clasif-predicted_values)
      tabla = pd.DataFrame(np.hstack((clasif, probas, real)), 
                          columns=['Clasificación Red', 'Probabilidad', 'Clasificación Real'])

      valores_unicos = tabla['Clasificación Real'].unique()
      nombres = names
      mapeo = {valor: nombre for valor, nombre in zip(valores_unicos, nombres)}
      tabla['Clasificación Red'] = tabla['Clasificación Red'].replace(mapeo)
      tabla['Clasificación Real'] = tabla['Clasificación Real'].replace(mapeo)

      return tabla
