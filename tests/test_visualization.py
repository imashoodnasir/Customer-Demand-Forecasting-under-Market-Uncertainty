from bayesian_retail.statistical.cd import critical_difference

def test_cd():
    assert critical_difference(5,10)>0
