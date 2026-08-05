import numpy as np, pandas as pd
from bayesian_retail.forecasting.aggregation import build_aggregation_matrix, aggregate_posterior_samples

def test_aggregation():
    h=pd.DataFrame({"series_id_idx":[0,1,2],"series_id":["a","b","c"],"category_id":["x","x","y"]})
    agg=build_aggregation_matrix(h,"category")
    samples=np.ones((2,3,4)); out=aggregate_posterior_samples(samples,agg)
    assert out.shape==(2,2,4); assert np.all(out[:,0,:]==2)
