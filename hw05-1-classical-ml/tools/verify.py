from pathlib import Path
import sys,json
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from algorithms import *
checks=[]
def check(name,ok):
 assert ok,name
 checks.append(name)
x=np.linspace(-1,1,100).reshape(-1,1);y=3*x+2
m=LinearRegressionGD(lr=.1,n_iters=2000);m.fit(x,y.ravel())
least_squares=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0].ravel()
check('linear GD agrees with least squares',np.allclose(np.r_[m.weight.ravel(),m.bias],least_squares,atol=1e-6))
check('linear prediction column shape',m.predict(x).shape==y.shape)
check('MSE handles mixed vector and column targets',mean_squared_error(y,y.ravel())==0)
check('constant perfect R2',r2_score([2,2],[2,2])==1)
check('constant imperfect R2',r2_score([2,2],[3,3])==0)
l=LogisticRegressionGD()
with np.errstate(over='raise',invalid='raise'):
 proba=l.sigmoid(np.array([-1000,0,1000]))
check('stable sigmoid at extreme logits',np.allclose(proba,[0,.5,1]))
X=np.array([[-2],[-1],[1],[2]],float);labels=np.array([0,0,1,1]);l.fit(X,labels)
check('logistic learns separable example',np.array_equal(l.predict(X),labels))
check('precision zero denominator',precision_score([0,1],[0,0])==0)
check('recall zero denominator',recall_score([0,0],[1,0])==0)
cm=confusion_matrix([0,0],[0,1])
check('confusion matrix includes prediction-only classes',np.array_equal(cm,[[1,1],[0,0]]))
t=DecisionTree(max_depth=3);t.fit(X,np.array([3,3,7,7]))
check('tree supports noncontiguous class labels',np.array_equal(t.predict(X),[3,3,7,7]))
t.fit(np.ones((4,1)),np.array([0,0,0,1]))
check('constant-feature tree chooses majority leaf',np.array_equal(t.predict(np.ones((2,1))),[0,0]))
a=np.arange(20).reshape(-1,1);b=np.arange(20)
train,test,yt,yv=train_test_split_numpy(a,b)
check('split is disjoint and complete',set(train.ravel()).isdisjoint(test.ravel()) and set(np.r_[train.ravel(),test.ravel()])==set(range(20)))
check('split preserves feature target alignment',np.array_equal(train.ravel(),yt) and np.array_equal(test.ravel(),yv))
check('split seed reproduces arrays',all(np.array_equal(i,j) for i,j in zip(train_test_split_numpy(a,b),train_test_split_numpy(a,b))))
Xmo,y=make_moons_numpy();check('moons generator shape and balance',Xmo.shape==(300,2) and np.array_equal(np.bincount(y),[150,150]))
result={'checks_passed':len(checks),'checks':checks}
(root/'results/verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
