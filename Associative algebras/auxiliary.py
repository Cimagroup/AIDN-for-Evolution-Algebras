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

    x=Dense(10*input_dim+2, use_bias=bias,activation=activation_function)(inputs)
    # More layers can be added here
    predictions=Dense(input_dim,use_bias=bias, activation=activation_function ,name='final_output')(x)



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
        i1 = d**2
        i2 = (d**2-d)*d
        i3 = int(d^2*(d-1))
        Ar2 = tf.slice(y_pred,[1],[i2])
        Ar3 = tf.slice(y_pred,[i3],[i1])
        A2=K.mean(math_ops.square(Ar2), axis=0)
        A3=K.mean(math_ops.square(Ar3), axis=0)
        A=A2+A3
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

    model.compile(optimizer= keras.optimizers.RMSprop(learning_rate=lr), loss = lossfunction  )           
    
    history=model.fit(x_data, y_data,  batch_size=batch_size, epochs=epochs, shuffle = True,  verbose=1) 
    
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
