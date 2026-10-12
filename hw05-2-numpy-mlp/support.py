"""New portfolio replacements for missing course utils; not original scaffold."""
from pathlib import Path
import copy
import numpy as np

def eval_numerical_gradient(f,x,verbose=False,h=1e-5):
    gradient=np.zeros_like(x,dtype=float)
    for index in np.ndindex(x.shape):
        original=x[index]
        try:
            x[index]=original+h;positive=f(x)
            x[index]=original-h;negative=f(x)
        finally:x[index]=original
        gradient[index]=(positive-negative)/(2*h)
    return gradient

def eval_numerical_gradient_array(f,x,df,h=1e-5):
    return eval_numerical_gradient(lambda value:float(np.sum(f(value)*df)),x,h=h)

def load_housing():
    with np.load(Path(__file__).resolve().parent/'data/california_housing.npz',allow_pickle=False) as bundle:
        return bundle['X'].copy(),bundle['y'].copy(),bundle['feature_names'].tolist()

def split_and_normalize(X,y,seed=42):
    order=np.random.default_rng(seed).permutation(len(X))
    n_train=int(.7*len(X));n_val=int(.15*len(X))
    split={'train':order[:n_train],'val':order[n_train:n_train+n_val],'test':order[n_train+n_val:]}
    mean=X[split['train']].mean(axis=0);std=X[split['train']].std(axis=0);std=np.where(std==0,1,std)
    y_mean=float(y[split['train']].mean());y_std=float(y[split['train']].std()) or 1.
    data={}
    for name,indices in split.items():
        data['X_'+name]=(X[indices]-mean)/std
        data['y_'+name]=(y[indices].astype(float)-y_mean)/y_std
    stats={'X_mean':mean,'X_std':std,'y_mean':y_mean,'y_std':y_std,'indices':split}
    return data,stats

class Solver:
    def __init__(self,model,data,update_rule,optim_config=None,lr_decay=.95,num_epochs=20,batch_size=200,seed=42):
        self.model,self.data,self.update_rule=model,data,update_rule
        self.configs={key:copy.deepcopy(optim_config or {}) for key in model.params}
        self.lr_decay,self.num_epochs,self.batch_size=lr_decay,num_epochs,batch_size
        self.rng=np.random.default_rng(seed)
        self.loss_history=[];self.train_mse_history=[];self.val_mse_history=[]
        self.best_epoch=0;self.best_val_mse=float('inf')
    def mse(self,split):
        pred=self.model.loss(self.data['X_'+split]).reshape(-1)
        return float(np.mean((pred-self.data['y_'+split])**2))
    def train(self):
        X,y=self.data['X_train'],self.data['y_train']
        self.best_params=copy.deepcopy(self.model.params)
        for epoch in range(self.num_epochs):
            order=self.rng.permutation(len(X))
            for start in range(0,len(X),self.batch_size):
                indices=order[start:start+self.batch_size]
                loss,gradients=self.model.loss(X[indices],y[indices])
                if not np.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
                self.loss_history.append(float(loss))
                for key in self.model.params:
                    self.model.params[key],self.configs[key]=self.update_rule(self.model.params[key],gradients[key],self.configs[key])
            train_mse,val_mse=self.mse('train'),self.mse('val')
            self.train_mse_history.append(train_mse);self.val_mse_history.append(val_mse)
            if val_mse<self.best_val_mse:
                self.best_val_mse=val_mse;self.best_epoch=epoch+1;self.best_params=copy.deepcopy(self.model.params)
            for config in self.configs.values():config['learning_rate']*=self.lr_decay
            print(f'Epoch {epoch+1}: train MSE={train_mse:.5f}, validation MSE={val_mse:.5f}')
        self.model.params=copy.deepcopy(self.best_params)
