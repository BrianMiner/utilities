from __future__ import division
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import dill
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#generate the grid automatically

def generate_grid(X, pred_vars, resolution):
        
        if len(pred_vars) ==1:
            min1=X[pred_vars[0]].min()
            max1=X[pred_vars[0]].max()
            grid=[[a]  for a in np.linspace(min1,max1,resolution[0])]
        
        if len(pred_vars) ==2:
            min1=X[pred_vars[0]].min()
            max1=X[pred_vars[0]].max()
            min2=X[pred_vars[1]].min()
            max2=X[pred_vars[1]].max()
            grid=[[a,b]  for a in np.linspace(min1,max1,resolution[0]) for b in np.linspace(min2,max2,resolution[1])] 
        return(grid)

    
#calculate the per grid value prediction
def create_pdp(X, model, pred_vars, grid, type, proba, orig, osp, values,xgbDmatrix):

    
    
    X_hold=X.copy()

    for i in range(len(pred_vars)):
        X_hold[pred_vars[i]]=grid[i]

    if values:
        X_hold = X_hold.values
    
    if xgbDmatrix:
        X_hold = xgb.DMatrix(X_hold,np.zeros(X_hold.shape[0]),np.nan)
    
    if type=="regression":
        pred_value = model.predict(X_hold)

    if type=="classification":

        if proba:
            pred_value = model.predict_proba(X_hold)[:,1]
            pred_value=1.0/(1+(1/orig-1)/(1/osp-1)*(1/pred_value-1))

        else:
            pred_value = model.predict(X_hold)
            pred_value=1.0/(1+(1/orig-1)/(1/osp-1)*(1/pred_value-1))

    return(np.mean(pred_value))


#MAIN FUNCTION
def partial_dependency(X, model, pred_vars, grid=None, resolution=[50], type="regression", 
                       proba=True, parallel =False, n_jobs=1, verbose =0, orig=2, osp=2,
                       temp_folder="/tmp", plot=False,save_path=None, returnfig=True,values=False, xgbDmatrix=False ):
    """
    X= Training data used to construct the model (as Pandas DF)
    
    model = The model object (any that implements predict or predict_proba functions)
    
    pred_vars = Variable names of the variables of interest - have matching column names in X (as List)
    
    grid= The combinations of variables of interest (as list of lists) or NONE (automatic generation will be done)
    
    resolution = number of data points to create grid between min and max of X[predvar] (list - same length as predvar)
        
    type = 'regression' or 'classification'
    
    proba = bool if predict should be predict_proba()
    
    parallel = bool if should run in parallel
    
    n_jobs = number of jobs for Joblib Parallel
    
    verbose = verbosity for Joblib Parallel
    
    orig= original population proportion of '1's' for classification (leave as is if no oversampling)
    
    osp= oversampled proportion of '1's' for classification (leave as is if no oversampling)
    
    temp_folder = location for Joblib to save pickeled objects
    
    plot= (bool) for if a plot should be generated
    
    save_path = location to save the resulting plot to disk
    
    returnfig (bool) if should print the plot
    
    values (bool) prediction function requires numpy and not pandas (call X.values before predict)
    
    xgbDmatrix (bool) prediction function requires Dmatrix 
    
    
    RETURNS:
    A pandas df of the grid and the response values
    
    """
    

    os.environ["JOBLIB_TEMP_FOLDER"] = temp_folder
    
    if len(pred_vars)>2:
        raise ValueError('Can only pass in a maximum of 2 variables')
    
    
    #if grid is none, generate it
    if grid ==None:
        
        grid = generate_grid (X,pred_vars,resolution)
    
    else: #create from lists
        if len(pred_vars) ==1:
            grid=[[a] for a in grid]
        else:
            grid=[[a,b]  for a in grid[0] for b in grid [1]]
                
    
    
    
    
    if parallel:
        effects=Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(create_pdp)(X=X, model=model, pred_vars=pred_vars, grid= x, type=type,proba=proba, orig=orig, osp=osp, values=values,xgbDmatrix=xgbDmatrix) for x in grid)
    
    else:
        effects=[create_pdp(X=X, model=model, pred_vars=pred_vars, grid= x, type=type,proba=proba, orig=orig, osp=osp,values=values,xgbDmatrix=xgbDmatrix) for x in grid]
    
            
    grid_pd=pd.DataFrame(grid,columns=pred_vars)
    grid_pd['response']=effects
    
    if plot:
        
        if len(pred_vars) ==1:
            
            g=X[pred_vars].describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
            min_max_deciles=g.loc[['min','10%','20%','30%','40%','50%','60%','70%','80%','90%','max']].values.ravel()
        
            fig, ax = plt.subplots(figsize=(17, 10))

            ax.plot(grid_pd[pred_vars],grid_pd.response)
            ax.plot(min_max_deciles, np.zeros(min_max_deciles.shape[0]), 'r|', ms=7)  # rug plot
            ax.set_xlabel(pred_vars[0])
            ax.set_ylabel("Estimated Mean Response")
            ax.set_title("Partial Depency Plot")
        
        
        if len(pred_vars) ==2:
            fig = plt.figure(figsize=(17,10))
            ax = fig.gca(projection='3d')
            
            surf=ax.plot_trisurf(grid_pd[pred_vars[0]].values, grid_pd[pred_vars[1]].values, grid_pd['response'].values, cmap=cm.jet, linewidth=0)
            ax.set_xlabel(pred_vars[0])
            ax.set_ylabel(pred_vars[1])
            ax.set_zlabel('Average Response')
            ax.set_title("Partial Depency Plot")
        
        if save_path != None:
            
            plt.savefig(save_path+pred_vars[0]+'.png')
    
        if returnfig == True:
            plt.show() 
        
    return(grid_pd) 
