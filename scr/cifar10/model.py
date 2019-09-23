from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, SeparableConv2D,Multiply, Add, Lambda, MaxPooling2D,Concatenate,ZeroPadding2D,Permute,BatchNormalization,Activation,Lambda,AveragePooling2D,Concatenate,Dropout,Multiply,RNN,Embedding,Reshape,LSTM,TimeDistributed,Dropout
from keras.backend import identity as Identity       
from keras import regularizers

import numpy as np
import tensorflow as tf


LAYERS = 15


def conv_f(num, kernel_size,strides_size, regualr,connect):    
    return Conv2D(num, kernel_size, strides=strides_size, padding='same',kernel_initializer="he_normal",bias_initializer='zeros',kernel_regularizer=regularizers.l2(regualr))(connect) 

def conv_s(num, kernel_size,strides_size, regualr,connect):    
    return SeparableConv2D(num, kernel_size, strides=strides_size, padding='same',kernel_initializer="he_normal",bias_initializer='zeros',kernel_regularizer=regularizers.l2(regualr))(connect) 

def model_fn(actions, addFilter):
    skip_list =[]

    for i in actions:
        skip_list.append(i[3])
    connect_list = []

    for k in range(len(actions)):
        skip_rate = 5
        a = np.where(abs(skip_list - skip_list[k])<skip_rate)[0]
        connect_list.append(a[a!= k])
    print(connect_list)
    ip = Input(shape=(32, 32, 3))   
    x = ip
    k = 0

    for i in actions:
        morefilter = 10
        morefilter_add = addFilter#100#150
        number_filter = i[2]*morefilter+morefilter_add
        if i[1] <15:
            k_size = 3
        else:
            k_size = 5
        regualr = 0.0001


        if k==0:
            x = conv_f(number_filter, (k_size, k_size),(1,1),regualr, x)
            x = BatchNormalization()(x)
            print(k,i[0],'con',number_filter,k_size)
        else:
            if i[0]<5:    
                x = Activation('relu')(x)                 
                x = conv_s(number_filter, (1, 1),(1,1),regualr, x)
                x = BatchNormalization()(x)                        
                x = Activation('relu')(x) 
                x = conv_s(number_filter, (k_size, k_size),(1,1),regualr, x)
                x = BatchNormalization()(x)  
                x = Activation('relu')(x)                 
                x = conv_s(number_filter, (1, 1),(1,1),regualr, x)
                x = BatchNormalization()(x)                        
                x = Activation('relu')(x) 
                x = conv_s(number_filter, (k_size, k_size),(1,1),regualr, x)
                x = BatchNormalization()(x)                      
                print(k,i[0],'sep',number_filter,k_size)
            elif 5<=i[0]<20:
                x = Activation('relu')(x)                 
                x = conv_f(number_filter, (1, 1),(1,1),regualr, x)
                x = BatchNormalization()(x)                        
                x = Activation('relu')(x)                 
                x = conv_f(number_filter, (k_size, k_size),(1,1),regualr, x)
                x = BatchNormalization()(x)    
                print(k,i[0],'con',number_filter,k_size)

            elif 20<=i[0]<25:                           
                x = MaxPooling2D(pool_size=(2, 2),padding='same')(x)                       

                print(k,i[0],'max',number_filter,k_size)
            else:                    
                x = AveragePooling2D(pool_size=(2, 2),padding='same')(x)

                print(k,i[0],'avg',number_filter,k_size)


                
                
        #skip connection
        locals()['short_cut_%s'%k] = x

        if any(connect_list[k]<k): 
                connect_name = connect_list[k][connect_list[k]<k]
                if len(connect_name)>0:
                    concat_list = []
                    for w in range(len(connect_name)):
                        n = connect_name[w]
                        short_cut = locals()['short_cut_%s'%n]
                        x_shape = int(int(short_cut.shape[-1])/morefilter)+int(morefilter_add/morefilter)

                        stride_size = np.ceil(int(short_cut.shape[1])/int(x.shape[1]))                                
                        short_cut  = conv_f(x_shape, (1, 1),(stride_size,stride_size),regualr, short_cut)  
                        short_cut = BatchNormalization()(short_cut)                                           
                        short_cut = Activation('relu')(short_cut)                             
                        concat_list.append(short_cut)
                    concat_list.append(x)
                    x = Concatenate(axis=3)(concat_list)

        if  k < LAYERS:
            k+=1        

    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax',kernel_initializer="he_normal",bias_initializer='zeros',kernel_regularizer=regularizers.l2(regualr))(x)
    model = Model(ip, x)

    return model

