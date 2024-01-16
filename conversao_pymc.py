import pymc3 as pm
import numpy as np
import arviz as az

# Dados de evidências de conversão
conversions = np.array([1, 0, 1, 1])

# Definindo o modelo usando o PyMC3
def using_pymc():
  with pm.Model() as model:
      # Priori uniforme representada pela distribuição Beta
      prior = pm.Beta('prior', alpha=1, beta=1)
      
      # Likelihood binomial
      likelihood = pm.Bernoulli('likelihood', p=prior, observed=conversions)
      
      # Amostragem do posterior usando MCMC
      trace = pm.sample(1000)
      return trace

trace = using_pymc()
# Obtendo os resultados
posterior_samples = trace['prior']

# Calculando a probabilidade a posteriori
posterior_probability = (posterior_samples > 0.5).mean()

print("Probabilidade a posteriori:", posterior_probability)
az.plot_posterior(trace, var_names='prior', credible_interval=0.95, rope=[0.45, 0.55])
