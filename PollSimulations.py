import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
from pymc.Matplot import plot as mcplot

# Data
# size con lab ukip lib oth
polls = np.array([[1049, 0.4, 0.39, 0.05, 0.08, 0.09],
		 [2038, 0.47, 0.35, 0.04, 0.08, 0.5],
		 [1046, 0.45, 0.4, 0.02, 0.07, 0.05]])

resp = (polls[:,0]*polls[:,1:].T).T.astype(int)

size = resp.sum(axis=1)

Dir = pm.Dirichlet('Dir', theta=[1,1,1,1,1])
Mult = pm.Multinomial('Mult', 
						value=resp, 
						n=size, 
						p=Dir, 
						observed=True)

model = pm.Model([Dir, Mult])
mcmc = pm.MCMC(model)
mcmc.sample(iter=500000, 
			burn=50000, 
			thin=50)

mcplot(mcmc)
plt.show()
