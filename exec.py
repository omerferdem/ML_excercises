from MLregressor import MLregressor
from PreProcessor import PreProcess
import pandas as pd

prep = PreProcess(location='xs.csv',test_size=0.15)
Xtrain, Xtest, Ytrain, Ytest=prep.give_data()
regr = MLregressor(Xtrain,Ytrain)
lr_model = regr.run_lr()
lr_pred=lr_model().predict(Xtest)

print(1)