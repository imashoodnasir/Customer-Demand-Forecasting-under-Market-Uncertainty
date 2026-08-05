import numpy as np
from bayesian_retail.metrics.deterministic import rmse,mae,smape
from bayesian_retail.metrics.probabilistic import crps_ensemble,picp

def test_metrics():
    y=np.array([1.,2.]); p=np.array([1.,3.]); assert rmse(y,p)>0; assert mae(y,p)==.5; assert smape(y,p)>0
    s=np.array([[.5,1,1.5],[1.5,2,2.5]]); assert crps_ensemble(y,s)>=0; assert picp(y,[0,1],[2,3])==1
