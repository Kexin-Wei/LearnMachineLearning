# %%

import torch
from torch.distributions.multivariate_normal import MultivariateNormal

n=4
logstd = torch.zeros(n)
mean   = torch.Tensor([3,2,1,4])
dist = MultivariateNormal(mean,covariance_matrix=torch.diag(1*torch.exp(logstd)))
# %%
#dist.sample()
dist.entropy()
# %%
