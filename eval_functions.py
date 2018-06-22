def eval_model_results_binary (result_tbl, bins=10, return_gains = True):
    
    #results table: numpy array with shape (?,2). First col is actual, second is prediction
    #bins: number of bins for gains table
    
    #return_gains = True means return the gains tables
    
    #requires:
    import pandas as pd
    pd.options.display.float_format = '{:,.6f}'.format
    from sklearn.metrics import roc_auc_score
    import matplotlib.pyplot as plt
    import numpy as np
    

    #calculate gains / lift
    
    result=pd.DataFrame(result_tbl,columns=['actual','pred'])
    result['decile']=(bins)-(pd.qcut(result.pred,bins,labels=False))
    grp_dec=result.groupby('decile')
    mean_act_pred=grp_dec['actual','pred'].mean()
    tbl_gains=grp_dec['actual','pred'].agg(['count','sum', 'mean', 'min', 'max']).sort_values([('pred', 'mean')], ascending=False)
    
    l=pd.DataFrame(tbl_gains)
    l_actual=l['actual'].copy().reset_index()
    l_actual=l_actual.drop(['min','max'],axis=1)
    l_actual=l_actual.rename(columns={"mean": "Actual Response Rate","count": "Count","sum": "Responders","decile":"Decile"})
    l_pred=l['pred'].copy().reset_index()
    l_pred=l_pred.drop(['decile','count','sum'],axis=1)
    l_pred=l_pred.rename(columns={"mean": "Predicted Response Rate", "min":"Min Predicted", "max":"Max Predicted"})
    l_comb=pd.concat([l_actual,l_pred],axis=1)
        
    l_comb["Lift"]=l_comb['Actual Response Rate'] / result_tbl[:,0].mean()
    l_comb["Cume Responders"] = l_comb['Responders'].cumsum().astype(int)
    l_comb["Cume Count"] = l_comb['Count'].cumsum().astype(int)
    l_comb["Responders"]=l_comb["Responders"].astype(int)
    l_comb["Cume Response Rate"] = l_comb["Cume Responders"]/l_comb["Cume Count"]
    l_comb["Cume Lift"]=l_comb["Cume Response Rate"] / result_tbl[:,0].mean()
    
    l_comb['cum bads']=l_comb['Cume Count']-l_comb['Cume Responders']
    
    
    total_bads=l_comb['cum bads'].values[bins-1]
    total_goods=l_comb['Cume Responders'].values[bins-1]
    
    l_comb['per bads']= l_comb['cum bads'] / total_bads
    l_comb['per goods']= l_comb['Cume Responders'] / total_goods
    
    l_comb['KS']=np.abs(l_comb['per goods'] -l_comb['per bads'])
    
    
    
        
    
    print('AUC is: {}'.format(roc_auc_score(result_tbl[:,0],result_tbl[:,1])))
    print('KS: {}'.format(l_comb['KS'].values.max()))
    print('Total Responders: {}'.format(result_tbl[:,0].sum()))  
      
    print('Lift Bin 1 (Vs Mean): {}'.format(l_comb['Actual Response Rate'].values[0] / result_tbl[:,0].mean()))
    print('Lift Bin 1 (Vs Last Bin): {}'.format(l_comb['Actual Response Rate'].values[0] / l_comb['Actual Response Rate'].values[bins-1]))
   
  
    l_comb = l_comb[['Decile','Count','Responders','Actual Response Rate','Predicted Response Rate','Lift','Cume Count','Cume Responders','Cume Response Rate', 'Cume Lift','Min Predicted','Max Predicted']]
       


    #plot the mean actual and predicted gains table
    fig, ax = plt.subplots()
    ax.plot(mean_act_pred.actual)
    ax.plot(mean_act_pred.pred)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Actual Response Rate")
    ax.set_title("Avg Actual and Avg Prediction Performance")
    ax.legend()
    
    

    #just actual
    fig, ax = plt.subplots()
    ax.plot(mean_act_pred.actual)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Actual Response Rate")
    ax.set_title("Avg Actual Performance")
    
    #cum lift
    fig, ax = plt.subplots()
    ax.plot(l_comb['Cume Lift'].values)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Actual Cume Lift")
    ax.set_title("Actual Cume Lift Performance")
    
    if return_gains == True:
        
        return(l_comb)
    else:
        return (None)
    
def lift_decile_x_scorer (y_true, y_pred, lift_decile=1, proba=True):
    
    #requires:
    import pandas as pd
    import numpy as np
    
    
    if proba:
        y_pred=y_pred[:,1]
    
   #calculate gains / lift
    bins=10
    try:
        result=pd.DataFrame(pd.DataFrame({'actual':y_true,'pred':y_pred}))
        result['decile']=(bins)-(pd.qcut(result.pred,bins,labels=False))
        grp_dec=result.groupby('decile')
        mean_act_pred=grp_dec['actual'].mean()
        overall= y_true.mean() 
        return(mean_act_pred.iloc[lift_decile-1]/overall)
    except Exception , e:
        print (e)
        return(0)
    
    
    
def lift_decile_x (result_tbl, lift_decile):
    
    #requires:
    import pandas as pd
    import numpy as np

    #calculate gains / lift
    bins=10
    
    result=pd.DataFrame(result_tbl,columns=['actual','pred'])
    result['decile']=(bins)-(pd.qcut(result.pred,bins,labels=False))
    grp_dec=result.groupby('decile')
    mean_act_pred=grp_dec['actual','pred'].mean()
    tbl_gains=grp_dec['actual','pred'].agg(['count','sum', 'mean', 'min', 'max']).sort_values([('pred', 'mean')], ascending=False)
    
    l=pd.DataFrame(tbl_gains)
    l_actual=l['actual'].copy().reset_index()
    l_actual=l_actual.drop(['min','max'],axis=1)
    l_actual=l_actual.rename(columns={"mean": "Actual Response Rate","count": "Count","sum": "Responders","decile":"Decile"})
    l_pred=l['pred'].copy().reset_index()
    l_pred=l_pred.drop(['decile','count','sum'],axis=1)
    l_pred=l_pred.rename(columns={"mean": "Predicted Response Rate", "min":"Min Predicted", "max":"Max Predicted"})
    l_comb=pd.concat([l_actual,l_pred],axis=1)
        
    l_comb["Lift"]=l_comb['Actual Response Rate'] / result_tbl[:,0].mean()
    l_comb["Cume Responders"] = l_comb['Responders'].cumsum().astype(int)
    l_comb["Cume Count"] = l_comb['Count'].cumsum().astype(int)
    l_comb["Responders"]=l_comb["Responders"].astype(int)
    l_comb["Cume Response Rate"] = l_comb["Cume Responders"]/l_comb["Cume Count"]
    l_comb["Cume Lift"]=l_comb["Cume Response Rate"] / result_tbl[:,0].mean()
    
    return(l_comb["Cume Lift"].values[lift_decile -1])
    
def lift_deciles (result_tbl):
    
    #requires:
    import pandas as pd
    import numpy as np
    

    #calculate gains / lift
    bins=10
    
    result=pd.DataFrame(result_tbl,columns=['actual','pred'])
    result['decile']=(bins)-(pd.qcut(result.pred,bins,labels=False))
    grp_dec=result.groupby('decile')
    mean_act_pred=grp_dec['actual','pred'].mean()
    tbl_gains=grp_dec['actual','pred'].agg(['count','sum', 'mean', 'min', 'max']).sort_values([('pred', 'mean')], ascending=False)
    
    l=pd.DataFrame(tbl_gains)
    l_actual=l['actual'].copy().reset_index()
    l_actual=l_actual.drop(['min','max'],axis=1)
    l_actual=l_actual.rename(columns={"mean": "Actual Response Rate","count": "Count","sum": "Responders","decile":"Decile"})
    l_pred=l['pred'].copy().reset_index()
    l_pred=l_pred.drop(['decile','count','sum'],axis=1)
    l_pred=l_pred.rename(columns={"mean": "Predicted Response Rate", "min":"Min Predicted", "max":"Max Predicted"})
    l_comb=pd.concat([l_actual,l_pred],axis=1)
        
    l_comb["Lift"]=l_comb['Actual Response Rate'] / result_tbl[:,0].mean()
    l_comb["Cume Responders"] = l_comb['Responders'].cumsum().astype(int)
    l_comb["Cume Count"] = l_comb['Count'].cumsum().astype(int)
    l_comb["Responders"]=l_comb["Responders"].astype(int)
    l_comb["Cume Response Rate"] = l_comb["Cume Responders"]/l_comb["Cume Count"]
    l_comb["Cume Lift"]=l_comb["Cume Response Rate"] / result_tbl[:,0].mean()
    
    return(l_comb["Cume Lift"].values)  


def response_deciles (result_tbl):
    
    #requires:
    import pandas as pd
    import numpy as np

    #calculate gains / lift
    bins=10
    
    result=pd.DataFrame(result_tbl,columns=['actual','pred'])
    result['decile']=(bins)-(pd.qcut(result.pred,bins,labels=False))
    grp_dec=result.groupby('decile')
    mean_act_pred=grp_dec['actual','pred'].mean()
    tbl_gains=grp_dec['actual','pred'].agg(['count','sum', 'mean', 'min', 'max']).sort_values([('pred', 'mean')], ascending=False)
    
    l=pd.DataFrame(tbl_gains)
    l_actual=l['actual'].copy().reset_index()
    l_actual=l_actual.drop(['min','max'],axis=1)
    l_actual=l_actual.rename(columns={"mean": "Actual Response Rate","count": "Count","sum": "Responders","decile":"Decile"})
    l_pred=l['pred'].copy().reset_index()
    l_pred=l_pred.drop(['decile','count','sum'],axis=1)
    l_pred=l_pred.rename(columns={"mean": "Predicted Response Rate", "min":"Min Predicted", "max":"Max Predicted"})
    l_comb=pd.concat([l_actual,l_pred],axis=1)

    
    return(l_comb["Actual Response Rate"].values)




def eval_model_results_numeric (result_tbl, bins=10, return_gains = True):
    
    #results table: numpy array with shape (?,2). First col is actual, second is prediction
    #bins: number of bins for gains table
    
    #return_gains = True means return the gains tables
    
    #requires:
    import pandas as pd
    pd.options.display.float_format = '{:,.6f}'.format
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import numpy as np
    

    #calculate gains / lift
    
    result=pd.DataFrame(result_tbl,columns=['actual','pred'])
    result['decile']=(bins)-(pd.qcut(result.pred,bins,labels=False))
    grp_dec=result.groupby('decile')
    mean_act_pred=grp_dec['actual','pred'].mean()
    tbl_gains=grp_dec['actual','pred'].agg(['count','sum', 'mean', 'min', 'max', 'median']).sort_values([('pred', 'mean')], ascending=False)
    
    l=pd.DataFrame(tbl_gains)
    l_actual=l['actual'].copy().reset_index()
    l_actual=l_actual.drop(['min','max'],axis=1)
    l_actual=l_actual.rename(columns={"mean": "Average Response","count": "Count","sum": "Total","decile":"Decile", 'median' : 'Median Response'})
    l_pred=l['pred'].copy().reset_index()
    l_pred=l_pred.drop(['decile','count','sum'],axis=1)
    l_pred=l_pred.rename(columns={"mean": "Predicted Average", "min":"Min Predicted", "max":"Max Predicted"})
    l_comb=pd.concat([l_actual,l_pred],axis=1)
        
    l_comb["Lift"]=l_comb['Average Response'] / result_tbl[:,0].mean()
   
    
    
    
        
    
    print('R2 is: {}'.format(r2_score(result_tbl[:,0],result_tbl[:,1])))
      
    l_comb = l_comb[['Decile','Count','Average Response','Predicted Average','Min Predicted','Max Predicted', 'Median Response', 'Lift']]
       


    #plot the mean actual and predicted gains table
    fig, ax = plt.subplots()
    ax.plot(mean_act_pred.actual)
    ax.plot(mean_act_pred.pred)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Average Response")
    ax.set_title("Avg Actual and Avg Prediction Performance")
    ax.legend()
    
    
    #just actual
    fig, ax = plt.subplots()
    ax.plot(mean_act_pred.actual)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Average Response")
    ax.set_title("Avg Actual")

    
    #actual median
    fig, ax = plt.subplots()
    ax.plot(l_comb['Median Response'].values)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Median Response")
    ax.set_title("Median Actual")
    
    if return_gains == True:
        
        return(l_comb)
    else:
        return (None)
          



def eval_model_results_numeric_trimmed (result_tbl, bins=10, trim_prop=0.01 ,return_gains = True):
    
    #results table: numpy array with shape (?,2). First col is actual, second is prediction
    #bins: number of bins for gains table
    
    #return_gains = True means return the gains tables
    
    #requires:
    import pandas as pd
    pd.options.display.float_format = '{:,.6f}'.format
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import mstats

    #calculate gains / lift
    
    result=pd.DataFrame(result_tbl,columns=['actual','pred'])
    result['decile']=(bins)-(pd.qcut(result.pred,bins,labels=False))
    grp_dec=result.groupby('decile')
    mean_act_pred=grp_dec['actual','pred'].mean()
    tbl_gains=grp_dec['actual','pred'].agg(['count','sum', 'mean', 'min', 'max', 'median']).sort_values([('pred', 'mean')], ascending=False)
    

    from scipy.stats import mstats

    def winsorize_series(group,trim_prop):
        return mstats.winsorize(group, limits=[trim_prop,trim_prop]).mean()

    trimmed_mean=grp_dec['actual'].apply(winsorize_series,(trim_prop)).reset_index()
    


    l=pd.DataFrame(tbl_gains)
    l_actual=l['actual'].copy().reset_index()
    l_actual['Trimmed Mean'] = trimmed_mean['actual']      
    l_actual=l_actual.drop(['min','max'],axis=1)
    l_actual=l_actual.rename(columns={"mean": "Average Response","count": "Count","sum": "Total","decile":"Decile", 'median' : 'Median Response'})
    l_pred=l['pred'].copy().reset_index()
    l_pred=l_pred.drop(['decile','count','sum'],axis=1)
    l_pred=l_pred.rename(columns={"mean": "Predicted Average", "min":"Min Predicted", "max":"Max Predicted"})
    l_comb=pd.concat([l_actual,l_pred],axis=1)
        
    l_comb["Lift"]=l_comb['Average Response'] / result_tbl[:,0].mean()

    
    
    
        
    
    print('R2 is: {}'.format(r2_score(result_tbl[:,0],result_tbl[:,1])))
      
    l_comb = l_comb[['Decile','Count','Average Response','Predicted Average','Min Predicted','Max Predicted', 'Median Response','Trimmed Mean','Lift']]
       


    #plot the mean actual and predicted gains table
    fig, ax = plt.subplots()
    ax.plot(mean_act_pred.actual)
    ax.plot(mean_act_pred.pred)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Average Response")
    ax.set_title("Avg Actual and Avg Prediction Performance")
    ax.legend()
    
    
    #just actual
    fig, ax = plt.subplots()
    ax.plot(mean_act_pred.actual)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Average Response")
    ax.set_title("Avg Actual")

    
    #actual median
    fig, ax = plt.subplots()
    ax.plot(l_comb['Median Response'].values)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Median Response")
    ax.set_title("Median Actual")
    
    
    
    
    #actual trimmed Mean
    fig, ax = plt.subplots()
    ax.plot(l_comb['Trimmed Mean'].values)
    ax.set_xlabel("Predicted Score Bin")
    ax.set_ylabel("Trimmed Mean Response")
    ax.set_title("Avg Actual")
    
    if return_gains == True:
        
        return(l_comb)
    else:
        return (None)
