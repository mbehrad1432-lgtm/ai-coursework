from pathlib import Path
import sys,json
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from network import *
from support import *
checks=[];errors={}
def check(name,ok):
    assert ok,name
    checks.append(name)
rng=np.random.default_rng(42)
x=rng.normal(size=(3,2,2));w=rng.normal(size=(4,2));b=rng.normal(size=2);dout=rng.normal(size=(3,2))
_,cache=affine_forward(x,w,b);dx,dw,db=affine_backward(dout,cache)
for name,value,analytical in [('x',x,dx),('w',w,dw),('b',b,db)]:
 f={'x':lambda _:affine_forward(x,w,b)[0],'w':lambda _:affine_forward(x,w,b)[0],'b':lambda _:affine_forward(x,w,b)[0]}[name]
 numerical=eval_numerical_gradient_array(f,value,dout);error=float(np.max(np.abs(numerical-analytical)))
 errors['affine_'+name]=error;check('affine '+name+' finite difference',error<1e-7)
for name,forward,backward in [('relu',relu_forward,relu_backward),('sigmoid',sigmoid_forward,sigmoid_backward)]:
 value=rng.normal(size=(4,3));upstream=rng.normal(size=value.shape);_,cache=forward(value)
 gradient=backward(upstream,cache);numerical=eval_numerical_gradient_array(lambda _:forward(value)[0],value,upstream)
 check(name+' finite difference',np.allclose(gradient,numerical,atol=1e-7))
with np.errstate(over='raise',invalid='raise'):
 out,_=sigmoid_forward(np.array([-1000.,0.,1000.]));gradient=sigmoid_backward(np.ones(3),np.array([-1000.,0.,1000.]))
check('sigmoid extreme values stay finite',np.allclose(out,[0,.5,1]) and np.all(np.isfinite(gradient)))
for shape in [(1,1),(3,2)]:
 value=rng.normal(size=shape);target=rng.normal(size=shape);_,gradient=mse_loss(value,target)
 numerical=eval_numerical_gradient(lambda _:mse_loss(value,target)[0],value)
 check('MSE finite difference '+str(shape),np.allclose(numerical,gradient,atol=1e-7))
check('scalar-output column and vector targets agree',mse_loss(np.array([[1.],[2.]]),np.array([1.,2.]))[0]==0)
try:mse_loss(np.zeros((2,2)),np.zeros(2))
except ValueError:badshape=True
else:badshape=False
check('MSE rejects broadcast targets',badshape)
for reg in [0.,.1]:
 np.random.seed(7);model=FullyConnectedNet([3],input_dim=2,output_dim=2,weight_scale=.2,dtype=np.float64,reg=reg)
 X=rng.normal(size=(3,2));y=rng.normal(size=(3,2));loss,gradients=model.loss(X,y)
 for key in model.params:
  numerical=eval_numerical_gradient(lambda _:model.loss(X,y)[0],model.params[key])
  error=float(np.max(np.abs(numerical-gradients[key])));errors[f'network_reg{reg}_{key}']=error
  check(f'network finite difference reg={reg} {key}',error<1e-6)
w=np.array([1.,2.]);g=np.array([.3,-.2]);next_w,config=sgd_momentum(w,g,{'learning_rate':.1,'momentum':0.})
check('zero momentum reduces to SGD',np.allclose(next_w,w-.1*g))
next_w,config=sgd_momentum(w,g,{'learning_rate':.1,'momentum':.9,'velocity':np.array([.2,.1])})
check('momentum retains previous velocity',np.allclose(config['velocity'],.9*np.array([.2,.1])-.1*g))
X,y,names=load_housing();data,stats=split_and_normalize(X,y)
sets=[set(indices.tolist()) for indices in stats['indices'].values()]
check('splits are disjoint and complete',all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i+1,3)) and len(set.union(*sets))==len(X))
check('normalization fitted only on train',np.allclose(stats['X_mean'],X[stats['indices']['train']].mean(axis=0)))
check('targets preserve fractional values',np.allclose(data['y_train']*stats['y_std']+stats['y_mean'],y[stats['indices']['train']]))
np.random.seed(42);small=FullyConnectedNet([],input_dim=2,output_dim=1,dtype=np.float64)
X=rng.normal(size=(80,2));y=X[:,0]*2-X[:,1]
d={'X_train':X[:60],'y_train':y[:60],'X_val':X[60:],'y_val':y[60:]}
solver=Solver(small,d,sgd_momentum,{'learning_rate':.03,'momentum':.9},num_epochs=10,batch_size=20)
initial=solver.mse('train');solver.train()
check('solver learns a known linear function',solver.mse('train')<initial*.1)
check('solver restores best validation checkpoint',np.isclose(solver.mse('val'),min(solver.val_mse_history)))
result={'checks_passed':len(checks),'checks':checks,'max_abs_gradient_errors':errors}
(root/'results/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
