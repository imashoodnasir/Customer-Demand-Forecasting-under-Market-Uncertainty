import numpy as np
from bayesian_retail.forecasting.recursive import RecursiveForecaster

def test_recursive():
    def sampler(history,cov,h,rng): return history[:,-1]+1
    r=RecursiveForecaster(sampler,3,2,11).forecast(np.array([[1,2],[3,4]]),np.zeros((2,3,1)))
    assert r.samples.shape==(2,2,3); assert np.all(r.samples[0,0]==[3,4,5])
