from keras.models import Model
from keras.layers import Conv2D, Input, BatchNormalization

def model(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(filters=128,kernel_size=(9,9), kernel_initializer ='glorot_uniform', 
                     activation='relu', padding ='same', use_bias=True, input_shape=input_shape)(X_input)
#     X = BatchNormalization()(X)
    
    X = Conv2D(filters = 64, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True)(X)
#     X = BatchNormalization()(X)

    X = Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='same', use_bias=True)(X)
    
    
    model = Model(inputs=X_input, outputs=X, name='SRCNN model')

    return model