from bayesian_retail.baselines.registry import registered_models

def test_registry(): assert registered_models()==['deepar','nbeats','prophet']
