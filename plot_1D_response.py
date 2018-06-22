import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, IntSlider, Layout
import ipywidgets as widgets
from IPython.display import display
import pandas as pd

class plot_1D_response:
    
   
    def __init__(self, df, target_var, pred_list, type_list, expand_target, num_bins_cont,no_print):
        

        
        self.df=df
        self.target_var =target_var
        self.pred_list = pred_list
        self.expand_target = expand_target
        self.type_lst = type_list
        self.num_bins_cont = num_bins_cont
        self.no_print=no_print
                
        self.hold_plots ={}
        self.hold_tables ={}
        
        if expand_target:
            self.y=self.expand_cat_target(df[self.target_var])
        else:
            self.y=df[self.target_var]
            
        
        
    def expand_cat_target (self,target):
        return(pd.get_dummies(target))
    
    def summarize(self,binned, var_name):
        binned=pd.concat([binned,self.y],axis=1)
        binned=binned.groupby(var_name).mean().reset_index()
        self.hold_tables[var_name]=binned
        return(binned)
    
    def plot_binned(self,sum_binned,var_name, sum_type):

        if sum_type=='num':
            xlabel=var_name + ' Bin'
            plot_type='line'
            stacked=False
        else:
            xlabel=var_name + ' Value'
            plot_type='bar'
            stacked=True
            
        ax=sum_binned.plot(x=var_name, figsize=(10,10),kind=plot_type, stacked=stacked)
        ax.set_title(var_name)
        ax.set_ylabel("Mean Response Value")
        ax.set_xlabel(xlabel)
        
        if self.no_print:
            plt.close()
        
        return(ax)
        
    def interact_plots(self):
        interact(self.print_plot, var_name=self.pred_list)
        
    def interact_tables(self):
        interact(self.print_table, var_name=self.pred_list)
        
        
    def bin_numeric (self, var_name):
        #return the column as bin with standard name
        temp_df= (self.num_bins_cont)-(pd.qcut(self.df[var_name],self.num_bins_cont,labels=False,duplicates='drop'))
        return(temp_df)
        
    def print_plot(self,var_name):
        ax=self.hold_plots[var_name]
        return(ax.get_figure())
     
    def print_table(self,var_name):
        tbl=self.hold_tables[var_name]
        return(tbl)       
    
    def create_plots(self):
        for indx, var in enumerate(self.pred_list):

            #if is numeric
            if self.type_lst[indx] == 'num' and self.df[var].nunique() >10:
                binned=self.bin_numeric(var)
            
            else:
                binned = self.df[var]
             
            sum_binned=self.summarize(binned, var) #summarized
            
            self.hold_plots[var]=self.plot_binned(sum_binned, var,self.type_lst[indx])
            print('Processed:',var)
            
