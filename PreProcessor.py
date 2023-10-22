import pandas as pd

class PreProcess:
    def __init__ (self,norm_in='Y',norm_out='Y',scale='MinMax',test_size=0.1):
        self.data=pd.read_csv('xs.csv')
    
    def give_data(self):
        final_data=0
        return final_data