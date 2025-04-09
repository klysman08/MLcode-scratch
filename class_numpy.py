# %%

import numpy 
x = numpy.random.uniform(0,5,250)

print(x)


import matplotlib.pyplot as plt
plt.hist(x,20)
plt.show()


# %%

import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(5,1,1000)
plt.hist(x,20)
plt.show()

# %%

import numpy as np
import matplotlib.pyplot as plt

x = np.random.normal(5,1,1000)
y = np.random.normal(10,2,1000)

plt.scatter(x,y)
plt.show()

# %%
