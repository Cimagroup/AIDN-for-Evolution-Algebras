from tensorflow  import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.python.ops import math_ops
from tensorflow.keras.layers import Concatenate
from tensorflow.linalg import matmul
from tensorflow import transpose
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint

def generator(input_dim=2,activation_function='sigmoid',bias=False):   
    """
    Purpose: 
    -------    
        
        This is a network that can used to represent element in a given presentation.          
    Arguments: 
    ---------    
        dim : integer dimension of the input
        activation_function : the activation function used in network.
        bias: determines if the network has bias. 
        * The choice of activation_function and bias determines the type of the representation.
         - When bias=False and activation_function='linear' the representation is linear.
         - When bias=True and activation_function='linear' the representation is linear.
         - When activation_function is not linear, the representation is not linear.
    Return: 
    -------    
        A neural network with the following properties :
        (1) input dimension=output dimension
        (2) used to represent an algebraic element.
          
    """


    inputs = Input(shape=(input_dim,))

    x=Dense(input_dim+2, use_bias=bias,activation=activation_function)(inputs)
    x=Dense(input_dim*254, use_bias=bias,activation=activation_function)(x)
    x=Dense(input_dim*128, use_bias=bias,activation=activation_function)(x)
    x=Dense(input_dim*32, use_bias=bias,activation=activation_function)(x)
    
    # More layers can be added here
    predictions=Dense(input_dim,use_bias=bias, activation="linear" ,name='final_output')(x)



    model = Model(inputs=inputs, outputs=predictions)
    
    return model

def get_n_operators(dim,activation_function,bias,n_of_operators):
    
    """
    Parameters: 
    ----------  
        dim : integer dimension of the input
        activation_function : the activation function used in network.
        bias: determines if the network has bias. 
        n_of_operators : number of neural network.
        
    Returns:
    --------    
        a list of n_of_operators networks. 
    """
    
    out=[generator(input_dim=dim,activation_function=activation_function,bias=bias) for i in range(0,n_of_operators)]

    return out



# Loss function trying to minimize the output of the group_rep_net
def group_rep_loss(input_dim=1,d=2):

    """
    Purpose
    -------
        
        Loss 
   
    Parameters
    ----------    
        input_dim, the dimension of the R_op generator.
        
    """
    
    def loss(y_true,y_pred):
        return K.mean(math_ops.square(y_pred),axis=0)
    
    return loss

def train_net(model,x_data,y_data,lossfunction,batch_size,lr,epochs,model_name="prueba_ass"):
    
    """
    Parameters: 
    ----------            
        model : keras model
        x_data : training X data
        y_data : training Y data
        model_name : name of the file where the model is going to be saved.
        lr: learning rate
        batch_size : batch size
        epochs : number of epochs
    
    """ 
    callback = keras.callbacks.EarlyStopping(patience = 1000,monitor='loss',restore_best_weights=True)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=1000,
        decay_rate=0.001,
        staircase=True)
    
    model.compile(optimizer= keras.optimizers.RMSprop(learning_rate=lr_schedule), loss = lossfunction  )   
    
    history=model.fit(x_data, y_data,  batch_size=batch_size, epochs=epochs, shuffle = True,  verbose=0,callbacks=[callback]) 
    
    return history

def get_relation_tensor(model,data):
    
    """
    Parameters:
    ----------    
        modelpath: folder where the model weights are located.
        model : keras model
        data : the data that we want to infer the model on.
    Returns:
    -------    
        the prediction of the input model on the input data.
            
    """
    

    return model.predict(data)

def check_matrix(matrix, tolerance=1e-1):
    # Check if the matrix is approximately symmetric
    if not np.allclose(matrix, matrix.T, atol=tolerance):
        return False
    
    # Check each row for the specified conditions
    for i in range(matrix.shape[0]):
        row = matrix[i, :]
        
        # Find indices of non-zero elements with tolerance
        non_zero_elements = row[np.abs(row) > 1e-1]
        
        # Condition 1: Only two non-zero elements in each row
        if len(non_zero_elements) != 2:
            return False
        
        # Condition 2: One of the non-zero elements should be on the diagonal
        if abs(matrix[i, i]) <= tolerance:
            return False
        
        # Condition 3: The two non-zero elements should approximately sum to zero
        if not np.isclose(np.sum(non_zero_elements), 0, atol=tolerance):
            return False
    
    return True
