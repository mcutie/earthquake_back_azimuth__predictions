#!/usr/bin/env python
# coding: utf-8


# Se importan las librerias necesarias

import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import pandas as pd
import math
from tensorflow.python.ops import math_ops
import datetime
import random
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

# Funcion para cargar los vectores de entrada y salida  de Training X1_train ,y_train_cos, y_train_sin

def load_training_data(load_path):
    X1_train = np.load(load_path + 'X1_train.npy')    
    y_train_cos = np.load(load_path + 'y_train_cos.npy') 
    y_train_sin = np.load(load_path + 'y_train_sin.npy')
    X1_test = np.load(load_path + 'X1_test.npy')    
    y_test_cos = np.load(load_path + 'y_test_cos.npy') 
    y_test_sin = np.load(load_path + 'y_test_sin.npy')     
    return X1_train, y_train_cos, y_train_sin, X1_test, y_test_cos, y_test_sin
    

# Funcion para crear el modelo ConvolucionalUsando con 1 entradas y para predecir dos salidas utilizando el API de Keras 
def create_model():
    input_1 = keras.Input(shape=(window_width,3),name="input_1")  
    
    # ----------------  Layer1 -------------------    
    x = layers.Conv1D(32,3, activation="relu",name="1_conv_layer_1")(input_1)
    x = layers.Dropout(0.2,name="1_dropout_1")(x)
    x = layers.MaxPooling1D(pool_size=2,name="1_max_pooling_1" )(x)
    
    # ----------------- Layer 2 ----------------
    x = layers.Conv1D(20,3, activation="relu",  name="1_conv_layer_2")(x)
    x = layers.Dropout(0.2,name="1_dropout_2")(x)
    x = layers.MaxPooling1D(pool_size=2,name="1_max_pooling_2" )(x)
    
    # ----------------- Layer 3 ----------------
    x = layers.Conv1D(20,3,activation="relu",name="1_conv_layer_3")(x)
    x = layers.Dropout(0.2,name="1_dropout_3")(x)
    x = layers.MaxPooling1D(pool_size=2,name="1_max_pooling_3" )(x)
    
    # ----------------- Layer 4 ----------------
    x = layers.Conv1D(20,3,activation="relu", name="1_conv_layer_4")(x)
    x = layers.Dropout(0.2,name="1_dropout_4")(x)
    x = layers.MaxPooling1D(pool_size=2,name="1_max_pooling_4" )(x)
    
    # ----------------- Layer 5 ----------------
    x = layers.Conv1D(20,3, activation="relu",name="1_conv_layer_5")(x)
    x = layers.Dropout(0.2,name="1_dropout_5")(x)
    x = layers.MaxPooling1D(pool_size=2,name="1_max_pooling_5" )(x)   
    
    
    x=layers.Flatten()(x)    
    layer_result = layers.Dense(64,activation ="relu")(x)
    layer_result = layers.Dropout(0.2,name="result_dropout_1")(layer_result)
    layer_result = layers.Dense(64,activation ="relu")(layer_result)
    layer_result = layers.Dropout(0.2,name="result_dropout_2")(layer_result)    
    cos = layers.Dense(1,name="cos")(layer_result)
    sin = layers.Dense(1,name="sin")(layer_result)    
    model = keras.Model(inputs=[input_1],outputs=[cos,sin],name="model_conv_two_output_one_branch")       
    return model

# Funcion para calcular el Coeficiente de Determinacion R2. Metrica usada en el entrenamiento
def R2(y_true, y_pred):    
    SS_res =  keras.backend.sum(keras.backend.square( y_true-y_pred )) 
    SS_tot = keras.backend.sum(keras.backend.square( y_true - keras.backend.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + keras.backend.epsilon()) )

    
# Programa principal

# Se cargan los datos 

data_load_path = "./Data_processed/SNR_03_azimuth_ranged_Kfold_1/"

X1_train, y_train_cos, y_train_sin, X1_test, y_test_cos, y_test_sin = load_training_data(data_load_path)
print('X1_train',X1_train.shape)
print('y_train_cos',y_train_cos.shape)
print('y_train_sin',y_train_sin.shape)
print('X1_test',X1_test.shape)
print('y_test_cos',y_test_cos.shape)
print('y_test_sin',y_test_sin.shape)

# Se crea un objeto de KFold
folds_quantity = 5
ramdom_seed = 7
kf = KFold(n_splits=folds_quantity, shuffle=True, random_state= ramdom_seed)

window_width = 500
fold_number = 1
epochs = 100000
batch_size = 32
patience = 100
learning_rate = 1.5e-5

losses = []  
cos_losses = []
sin_losses = []
cos_maes = []
sin_maes = []
cos_r2s = []
sin_r2s = []
fit_histories = []   


for train_index, test_index in kf.split(X1_train):
    keras.backend.clear_session()
    
    # Se separan los datos de entrenamiento y prueba para el fold actual
    
    X1_train_fold, X1_test_fold = X1_train[train_index], X1_train[test_index]
    y_train_sin_fold, y_train_cos_fold, y_test_sin_fold, y_test_cos_fold = y_train_sin[train_index],y_train_cos[train_index], y_train_sin[test_index], y_train_cos[test_index]
    print('Initializing Kfold %s'%str(fold_number))
    print('X1_train shape:',X1_train_fold.shape)
    print('X1_test shape:',X1_test_fold.shape)
    print('Output Sin Train shape:',y_train_sin_fold.shape)  
    print('Output Cos Train shape:',y_train_cos_fold.shape)  
    print('Output Sin Test shape:',y_test_sin_fold.shape)    
    print('Output Cos Test shape:',y_test_cos_fold.shape) 
   
    # Se crea el modelo de Keras para el fold actual
    model = create_model()
          
    # Compilamos nuestro modelo
    
    model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss={
      "cos": keras.losses.MeanSquaredError(),
      "sin": keras.losses.MeanSquaredError(),
    },    
    metrics=['mae',R2])
    

    # Path de los log para tensorboard
    log_dir = './Data_processed/SNR_03_azimuth_ranged_Kfold_1/logs/fit/folder' + str(fold_number) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    
    if not os.path.exists(log_dir):
      os.mkdir(log_dir)
    
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # patient early stopping
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience, restore_best_weights=True)

    # Entrena el modelo en el fold actual    
    history = model.fit({"input_1":X1_train_fold},{'cos':y_train_cos_fold, 'sin': y_train_sin_fold },epochs=epochs, batch_size=batch_size,validation_split=0.2, verbose=2, callbacks=[tensorboard,early_stopping])
            
    # Almacenar la historia de ajuste en la lista
    fit_histories.append(history)
    
   #Evaluando el modelo para el fold
    print('-----------------------------------------------------------')
    print("Evaluamos el modelo para el fold actual")
    
    #loss, cos_loss, sin_loss, cos_mae, cos_r2,  sin_mae, sin_r2  = model.evaluate({"input_1":X1_test_fold, "input_2": X2_test_fold},{'cos':y_test_cos_fold, 'sin': y_test_sin_fold })
    loss, cos_loss, sin_loss, cos_mae, cos_r2,  sin_mae, sin_r2  = model.evaluate({"input_1":X1_test_fold},{'cos':y_test_cos_fold, 'sin': y_test_sin_fold })
    print(' ')
    print('-----------------------------------------------------------')
    
    losses.append(loss)   
    cos_losses.append(cos_loss)
    sin_losses.append(sin_loss)
    cos_maes.append(cos_mae)
    sin_maes.append(sin_mae)
    cos_r2s.append(cos_r2)
    sin_r2s.append(sin_r2)    
    
    print("Metricas loss", loss)
    
    print('-------------------------------------------------------------')
    print("Metricas cos_loss", cos_loss)
    print("Metricas cos_mae", cos_mae)
    print("Metricas cos_r2", cos_r2)
    print('-------------------------------------------------------------')
    
    print("Metricas sin_loss", sin_loss)
    print("Metricas sin_mae", sin_mae)
    print("Metricas sin_r2", sin_r2)
    
    print('-------------------------------------------------------------')
    
    print("Media de loss", np.mean(losses))    
    print('-------------------------------------------------------------')
    print("Media cos_loss",np.mean(cos_losses))
    print("Media cos_mae", np.mean(cos_maes))
    print("Media cos_r2", np.mean(cos_r2s))
    print('-------------------------------------------------------------')
    print("Media sin_loss",np.mean(sin_losses))
    print("Media sin_mae", np.mean(sin_maes))
    print("Media sin_r2", np.mean(sin_r2s))
    print('-------------------------------------------------------------')
    
    #Evaluando el modelo con los valores de Validacion
    print('-----------------------------------------------------------')
    print("Evaluamos el modelo para los valores de Validacion")
    
    loss_val, cos_loss_val, sin_loss_val, cos_mae_val, cos_r2_val,  sin_mae_val, sin_r2_val  = model.evaluate({"input_1":X1_test},{'cos':y_test_cos, 'sin': y_test_sin })
    

    print(' ')
    print('-----------------------------------------------------------')
    print(' ')
    print("Metricas loss_val", loss_val)
    
    print('-------------------------------------------------------------')
    print("Metricas cos_loss_val", cos_loss_val)
    print("Metricas cos_mae_val", cos_mae_val)
    print("Metricas cos_r2", cos_r2_val)
    print('-------------------------------------------------------------')
    
    print("Metricas sin_loss_val", sin_loss_val)
    print("Metricas sin_mae_val", sin_mae_val)
    print("Metricas sin_r2_val", sin_r2_val)
    
    print('-------------------------------------------------------------')
    

    
    
    #Guardando el modelo 
    target_dir = './Data_processed/SNR_03_azimuth_ranged_Kfold_1/model_Azimuth_' + str(fold_number)
    if not os.path.exists(target_dir):
      os.mkdir(target_dir)
    model.save('./Data_processed/SNR_03_azimuth_ranged_Kfold_1/model_Azimuth_' + str(fold_number) + '/modelo.h5' )
    model.save_weights('./Data_processed/SNR_03_azimuth_ranged_Kfold_1/model_Azimuth_' + str(fold_number) +'/pesos.h5')    

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./Data_processed/SNR_03_azimuth_ranged_Kfold_1/model_Azimuth_' + str(fold_number) + '/loss.png')
    plt.close()

    # Se realizar predicciones en el conjunto de validación del fold    
    y_pred_cos_fold, y_pred_sin_fold = model.predict({"input_1":X1_test_fold})
    
    # Se realizar predicciones en el conjunto de validación    
    y_pred_cos, y_pred_sin = model.predict({"input_1":X1_test})
    
    # Salvamos los datos correspondientes al fold    
    
    np.save(target_dir + 'y_pred_cos_fold.npy',y_pred_cos_fold)
    np.save(target_dir + 'y_pred_sin_fold.npy',y_pred_sin_fold)
    
    # salvamos los datos correspondientes al test   

    
    np.save(target_dir + 'y_pred_cos.npy',y_pred_cos)
    np.save(target_dir + 'y_pred_sin.npy',y_pred_sin)
    
    fold_number +=1
print ('------------------------------------------------------------------------------------------------------------')





