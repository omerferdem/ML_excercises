import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class PostProcess:
    def __init__(self,Ynn,Ytest):
        self.Ynn=Ynn
        self.Ytest=Ytest
    
    def print_metrics(self):
        Ynn=self.Ynn
        Ytest=self.Ytest
        mae = mean_absolute_error(Ytest, Ynn)
        mse = mean_squared_error(Ytest, Ynn)
        r2 = r2_score(Ytest, Ynn)

        print('MAE: ', mae)
        print('MSE: ', mse)
        print('R2: ', r2)