import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json, load_model
from keras.engine.training import collect_trainable_weights
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Convolution1D, Convolution2D, Reshape, MaxPooling2D, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
K.set_learning_phase(1)

import ipdb

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, preprocess_state, vision):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size, preprocess_state, vision)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size, preprocess_state, vision)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in xrange(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim, preprocess_state, vision):
        print("Now we build the model")

        ff_structures = [HIDDEN1_UNITS, HIDDEN2_UNITS] if preprocess_state is False else [128, 256]
        S = Input(shape=[state_size])  
        A = Input(shape=[action_dim],name='action2')   

        if preprocess_state: I = self.state_pre_processing_net(S, vision)
        else : I = S

        V = self.connect_to_output_net(I, A, action_dim, ff_structures)

        model = Model(input=[S,A],output=V)
        model.summary()

        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 


    def state_pre_processing_net(self, S, vision):

        if vision is False:
            # split input
            S1 = Lambda(lambda x: x[:,:-19])(S) # physical input
            S2 = Lambda(lambda x: x[:,-19:])(S) # laser input
            
            # add thrid dimension to S2 in order to preform Conv
            C0 = Reshape((19,1))(S2)
            C1 = Convolution1D(4, 4, activation='relu')(C0)
            C2 = Convolution1D(4, 2, activation='relu')(C1)
            F1 = Flatten()(C2)

            # merge
            I = merge([S1,F1],mode='concat')
        
        else: # TODO: implement IMAGE feature extraction network here
            S1 = Lambda(lambda x: x[:,:-19-64*64*3])(S) # physical input
            S2 = Lambda(lambda x: x[:,-19-64*64*3:-64*64*3])(S) # laser input
            SV = Lambda(lambda x: x[:,-64*64*3:])(S)
            
            V0 = Reshape((64,64,3))(SV)
            V01 = BatchNormalization()(V0)
            V1 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(V01)
            V02 = BatchNormalization()(V1)
            V2 = MaxPooling2D(pool_size=(2, 2))(V02)
            V3 = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(V2)
            V4 = MaxPooling2D(pool_size=(2, 2))(V3)
            V5 = Convolution2D(128, 5, 5, border_mode='same', activation='relu')(V4)
            V06 = BatchNormalization()(V5)
            V6 = Convolution2D(256, 5, 5, border_mode='same', activation='relu')(V06)
            V7 = MaxPooling2D(pool_size=(2, 2))(V6)
            F2 = Flatten()(V7)
            
            # add thrid dimension to S2 in order to preform Conv
            C0 = Reshape((19,1))(S2)
            C1 = Convolution1D(4, 4, activation='relu')(C0)
            C02 = BatchNormalization()(C1)
            C2 = Convolution1D(4, 2, activation='relu')(C02)
            C3 = BatchNormalization()(C2)
            F1 = Flatten()(C3)

            #conv input (b_s, 64,64,3)#tf mode
            #Reshape( (64,64,3) , input_shape=(12288,))
            
            #import IPython
            #IPython.embed()


            # merge
            I = merge([S1,F1,F2],mode='concat')

        return I

    def connect_to_output_net(self, S, A, action_dim, ff_structures, useLSTM=False):
        """ ff_structures is the size hidden neurals for feed forward network """

        if not useLSTM:
            w1 = Dense(ff_structures[0], activation='relu')(S)
            a1 = Dense(ff_structures[1], activation='linear')(A) 
            h1 = Dense(ff_structures[1], activation='linear')(w1)
            h2 = merge([h1,a1],mode='sum')    
            h3 = Dense(ff_structures[1], activation='relu')(h2)
            V = Dense(action_dim,activation='linear')(h3)

        else: # TODO implement LSTM here
            pass

        return V
