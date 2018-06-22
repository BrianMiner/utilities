#called by callback at the end of each iteration (tree) - save just the best iteration based on cv error
class OOFCallback:
    def  __init__(self, oof_preds_dct, maximize=True):
        
        self.oof_preds_dct = oof_preds_dct
        self.best_eval =None
        self.maximize =maximize

    def __call__(self, env): #saved the env object and interogatted it to see what it contained
        
        #look at the current mean OOF eval
        current_eval = env.evaluation_result_list[1][1]
        
        if env.iteration % 10 ==0:
            print("Iteration #"+str(env.iteration))
        
        if self.best_eval == None: #first iteration
            self.best_eval = current_eval
            self.get_oof_preds(env.cvfolds)
        else:
            if self.maximize:
                if self.best_eval < current_eval:
                    self.best_eval = current_eval
                    #replace the actuals and preds
                    self.get_oof_preds(env.cvfolds)
            else:
                if self.best_eval > current_eval:
                    self.best_eval = current_eval
                    #replace the actuals and preds
                    self.get_oof_preds(env.cvfolds)
        
    def get_oof_preds(self, cvfolds):
        #reset the list of actuals and preds
        self.oof_preds_dct['actual']=[]
        self.oof_preds_dct['preds']=[]
        #loop through the folds
        for i, fold in enumerate(cvfolds):
            self.oof_preds_dct['actual'].extend(fold.dtest.get_label())
            self.oof_preds_dct['preds'].extend(fold.bst.predict(fold.dtest))
