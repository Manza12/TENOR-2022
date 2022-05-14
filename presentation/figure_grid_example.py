import numpy as np
import matplotlib.pyplot as plt

# Timestamps
timestamps = np.array([0, 1.018, 1.531, 2.061, 2.888, 3.179, 4.286])

# Plot
n = len(timestamps)
notes = np.zeros(n)

fig = plt.figure(figsize=(5., 1.))

ax = fig.gca()
points = plt.scatter(timestamps, notes, marker='|',  s=100)

ax.set_xlabel('Time (s)')
ax.get_yaxis().set_ticks([])
if not len(notes) == 0:
    ax.set_ylim([np.min(notes) - 2, np.max(notes) + 2])
else:
    ax.set_ylim([-0.1, 0.5])

plt.tight_layout()

fig.savefig('rhythm_ticks.svg', transparent=True)
plt.show()


if __name__ == '__main__':
    pass
