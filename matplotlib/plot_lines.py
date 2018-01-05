import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-np.pi, np.pi, 256,endpoint=True)
print x.shape
y = np.sin(x)


# configurations
plt.figure(figsize=(8,6), dpi=80)

plt.xlim(x.min()*1.1, x.max()*1.1)
plt.xticks( [-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.ylim(y.min()*1.1, y.max()*1.1)
plt.yticks([-1, 0, +1])

plt.subplot(221)
plt.plot(x,y,'r-')
plt.subplot(222)
plt.plot(1,2,'ro')


plt.show()
