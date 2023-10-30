import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

class MLregressor:
    def __init__ (self,Xtrain,Ytrain,rfr_trees=100,nn_layers=[100, 150, 30, 50],nn_lr=0.0009,nn_epochs=100,nn_batch_size=16,nn_validation_split=0.15):
        self.Xtrain=Xtrain
        self.Ytrain=Ytrain
        self.rfr_trees=rfr_trees
        self.nn_layers=nn_layers
        self.nn_lr=nn_lr
        self.nn_epochs=nn_epochs
        self.nn_batch_size=nn_batch_size
        self.nn_validation_split=nn_validation_split

    def run_lr(self):
        # Use model.predict(Xtest) to get output.
        def model():
            lr=LinearRegression()
            return lr.fit(self.Xtrain,self.Ytrain)
        return model()
    
    def run_rfr(self):
        # Use model.predict(Xtest) to get output.
        def model():
            rfr=RandomForestRegressor(self.rfr_trees)
            return rfr.fit(self.Xtrain,self.Ytrain)
        return model()

    def run_nn(self):
        # Use model.predict(Xtest) to get output.
        def model():
            #fitting neural network
            model = Sequential()
            num_dense_layers = len(self.nn_layers)
            num_nodes = self.nn_layers
            
            model.add(Dense(num_nodes[0], kernel_initializer='normal', activation='relu', input_dim=self.Xtrain.shape[1]))
            #model.add(Dropout(rate=0.5))
            for i in range(1, num_dense_layers):
                model.add(Dense(num_nodes[i], activation='relu', kernel_initializer='normal'))
            model.add(Dense(1, activation='linear', kernel_initializer='normal'))
            
            lr = self.nn_lr
            model.compile(loss='mean_absolute_error', optimizer=Adam(lr), metrics=['mean_absolute_error'])
            model.summary()
            
            checkpoint = ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.9, patience=5, min_lr=0, verbose=1)
            
            #save the best model to use for prediction
            model_call=ModelCheckpoint('mybest_nn.h5', monitor='val_mean_absolute_error', save_best_only=True, verbose=True)
            
            history = model.fit(self.Xtrain, self.Ytrain, epochs=self.nn_epochs, batch_size=self.nn_batch_size, validation_split=self.nn_validation_split, callbacks=[checkpoint, model_call])
            return model, history
        return model()