import numpy as np
from matplotlib import pyplot as plt

def likelihood(prior, evidence):
  output = 1
  for ev in evidence:
    if ev == '0': output *= (1.-prior)
    elif ev == '1': output *= (prior)

  return output

def proba_of_evidence(evidence, prior_belief):
  # print(f'initial:{initial}, likelihood:{likelihood(initial, evidence=evidence)}, product:{initial*likelihood(initial, evidence=evidence)}')
  output = 0
  for theta, p_theta in prior_belief():
    output += p_theta*likelihood(theta, evidence=evidence)
  return output

def posterior(prior_belief, evidence='101010'):
  output = []
  for theta, p_theta in prior_belief():
    posterior_value = (p_theta * likelihood(theta, evidence=evidence)) / proba_of_evidence(evidence=evidence, prior_belief=prior_belief)
    output.append(posterior_value)
  return output

# steps=1/6
steps = 0.01
possible_values = np.arange(0, 1, steps)
prior_proba = np.ones(possible_values.shape) * steps
# prior_proba = [.1, .8, .1, 0, 0, 0]
# prior_proba = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
prior_belief = lambda: zip(possible_values, prior_proba)
fig, ax = plt.subplots()
offset = steps/5
ax.plot(possible_values-offset, prior_proba, color='gray', label='prior')#, width=offset)

draws = '100000'
ax.plot(possible_values, posterior(prior_belief, evidence=draws), color='k', label='small_draws')#, width=offset)

draws = '100000'*2
ax.plot(possible_values+offset, posterior(prior_belief, evidence=draws), color='b', label='medium_draws')#, width=offset)

draws = '100000'*10
ax.plot(possible_values+offset*2, posterior(prior_belief, evidence=draws), color='r', label='large_draws')#, width=offset)

ax.set_xticks(np.arange(0, 1, steps))
plt.legend()
plt.show()