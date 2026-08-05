import torch
from bayesian_retail.baselines.autoformer.model import Autoformer
from bayesian_retail.baselines.fedformer.model import FEDformer
from bayesian_retail.baselines.timesnet.model import TimesNet

def test_models():
    x=torch.randn(2,64,4)
    for m in [Autoformer(4), FEDformer(4), TimesNet(4)]:
        y=m(x)
        assert y.shape[0]==2
