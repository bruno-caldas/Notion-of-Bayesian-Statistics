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
faces = 1000; steps=1/faces
possible_values = np.linspace(0, 1, faces)
# prior_proba = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
prior_proba = np.ones(possible_values.shape[0]) * 1/faces
prior_belief = lambda: zip(possible_values, prior_proba)
fig, ax = plt.subplots()
offset = steps/5
ax.bar(possible_values, prior_proba, color='gray', label='prior', width=offset)
ax.set_ylabel("probability")
ax.set_xlabel("proportion")
draws = '1011000010'
ax.plot(possible_values, posterior(prior_belief, evidence=draws), 'o', color='r', markersize=12, label='posterior')
ax.plot(possible_values, posterior(prior_belief, evidence=draws*2), 'o', color='b', markersize=12, label='posterior with more evidence')
ax.plot(possible_values, posterior(prior_belief, evidence=draws*20), 'o', color='g', markersize=12, label='posterior with much more evidence')

# draws = '100000'*2
# ax.bar(possible_values+offset, posterior(prior_belief, evidence=draws), color='b', label='medium_draws', width=offset)

# draws = '100000'*10
# ax.bar(possible_values+offset*2, posterior(prior_belief, evidence=draws), color='r', label='large_draws', width=offset)

# ax.set_xticks(possible_values)
# ax.set_xlim(0,1)
# ax.set_ylim(0,1)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# Put a legend to the right of the current axis
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plt.legend()
plt.show()