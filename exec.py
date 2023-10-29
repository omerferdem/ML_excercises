from PostProcessor import PostProcess
from MLregressor import MLregressor
from PreProcessor import PreProcess
import pandas as pd

prep = PreProcess(location='xs.csv',test_size=0.15)
Xtrain, Xtest, Ytrain, Ytest=prep.give_data()

regr = MLregressor(Xtrain,Ytrain)
lr_model = regr.run_lr()
lr_pred=lr_model().predict(Xtest)

postp = PostProcess(lr_pred,Ytest)
print('Linear Regression Metrics:')
postp.print_metrics()

# *************************************

rfr_model = regr.run_rfr()
rfr_pred=rfr_model().predict(Xtest)

postp = PostProcess(rfr_pred,Ytest)
print('Random Forest Regression Metrics:')
postp.print_metrics()

# *************************************

nn_model,nn_history = regr.run_nn()
nn_pred=nn_model.predict(Xtest)

postp = PostProcess(nn_pred,Ytest)
print('Neural Network Metrics:')
postp.print_metrics()