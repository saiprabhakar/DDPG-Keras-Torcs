import numpy as np
import math
from keras.initializations import normal, identity
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.engine.training import collect_trainable_weights
from keras.layers import Dense, Flatten, Input, merge, Lambda, Convolution1D, Convolution2D, Reshape, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
K.set_learning_phase(1)
import ipdb

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class ActorNetwork(object):
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LEARNING_RATE, preprocess_state=False, vision=False):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

        K.set_session(sess)

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size, preprocess_state, vision)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size, preprocess_state, vision) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(grads)
        self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in xrange(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim, preprocess_state, vision):
        print("Now we build the model")
        
        ff_structures = [HIDDEN1_UNITS, HIDDEN2_UNITS] if preprocess_state is False else [128, 256]
        S = Input(shape=[state_size])

        if preprocess_state: I = self.state_pre_processing_net(S, vision)
        else : I = S

        V = self.connect_to_output_net(I, action_dim, ff_structures)
        model = Model(input=S,output=V)
        model.summary()
        # 

        return model, model.trainable_weights, S

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

    def connect_to_output_net(self, S, action_dim, ff_structures, useLSTM=False):
        """ ff_structures is the size hidden neurals for feed forward network """

        if not useLSTM:
            h0 = Dense(ff_structures[0], activation='relu')(S)
            h1 = Dense(ff_structures[1], activation='relu')(h0)
            Steer = Dense(1, activation='tanh')(h1)
            if action_dim == 3:
                Acceleration = Dense(1,activation='sigmoid')(h1)
                Brake = Dense(1,activation='sigmoid')(h1)
                O = merge([Steer,Acceleration,Brake],mode='concat')  
            elif action_dim == 1:
                O = Steer

        else: # TODO implement LSTM here
            pass

        return O


