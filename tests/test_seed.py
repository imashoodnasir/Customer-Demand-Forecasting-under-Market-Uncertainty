import random
import numpy as np
from bayesian_retail.seed import seed_everything

def test_seed():
    seed_everything(123)
    a, b = random.random(), np.random.random()
    seed_everything(123)
    assert a == random.random()
    assert b == np.random.random()
