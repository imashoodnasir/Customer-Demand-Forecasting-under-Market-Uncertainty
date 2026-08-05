from bayesian_retail.experiments.runner import ExperimentRunner

def test_jobs():
    r=ExperimentRunner(["m5"],[11])
    assert len(r.run())==13
