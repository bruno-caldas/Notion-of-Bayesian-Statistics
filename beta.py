import numpy as np
from scipy.stats import beta
from matplotlib import pyplot as plt

def a(theta, n): return theta*n
def b(theta, n): return (1-theta) * n

x = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
ax.set_ylabel("probability")
ax.set_xlabel("proportion")

ax.plot(x, beta.pdf(x, a(theta=40/100, n=10),b(theta=40/100, n=10)), color='r', label='posterior')
ax.plot(x, beta.pdf(x, a(theta=40/100, n=20),b(theta=40/100, n=20)), color='b', label='posterior with more evidence')
ax.plot(x, beta.pdf(x, a(theta=40/100, n=100),b(theta=40/100, n=100)), color='g', label='posterior with much more evidence')

# alpha = 4
# beta = 6

# x = np.linspace(0, 1, 100) 
# y = (x**(alpha-1) * (1-x)**(beta-1))
# y_normalized = y / np.trapz(y, x)
# ax.plot(x,y)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()
