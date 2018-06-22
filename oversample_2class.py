def oversample_2class(x,y,prop,seed=0,save_msk_path=None, shuffle=True):
    
    y_pos=y.loc[(y.values).flatten()==1]
    x_pos =x.loc[(y.values).flatten()==1,:]
    
    nb_pos=y_pos.shape[0]
    nb_neg = (nb_pos / prop) - nb_pos
    
    ttl_neg=x.loc[(y.values).flatten()==0,:]
    np.random.seed(seed)
    indx_msk=np.random.choice(a=ttl_neg.shape[0], size=int(nb_neg),replace = False)
    
    sampled_negx=x.iloc[indx_msk,:]
    sampled_negy=y.iloc[indx_msk]
    
    finalx_df=pd.concat([x_pos,sampled_negx],axis=0)
    finaly_df=pd.concat([y_pos,sampled_negy],axis=0)
    
    
    if save_msk_path != None:
        joblib.dump(indx_msk,save_msk_path)
        
    
    if shuffle:
        indx_perm=np.random.permutation(finalx_df.shape[0])
        finalx_df=finalx_df.iloc[indx_perm,:]
        finaly_df=finaly_df.iloc[indx_perm,:]
        
        
        
    return(finalx_df,finaly_df)
