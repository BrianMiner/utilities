import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, IntSlider, Layout
import ipywidgets as widgets
from IPython.display import display
import pandas as pd

class freq_tables:
    
   
    def __init__(self, df, numeric_cut=20):
        

        
        self.df=df
        self. numeric_cut = numeric_cut
        self.hold_counts={}
        self.list_cols = []
        
    def create_freqs(self):
        for column in self.df:
            if (self.df[column].dtype == object) or (self.df[column].nunique()) <self.numeric_cut:
                self.list_cols.append(column)
                v1=self.df[column].value_counts()
                v1=v1.reset_index().rename(columns={"index": "Level", column: "Count"})
                v1=v1.sort_values('Level')
                v1=v1.append([{'Level':'NULL', 'Count':self.df[column].isnull().sum()}])
                v1=v1.reset_index(drop=True)
                self.hold_counts[column]=v1
        
    def display_table (self, var_name):
        return(self.hold_counts[var_name])

    def interact_tables (self):
        interact(self.display_table, var_name=sorted(self.list_cols)) 
