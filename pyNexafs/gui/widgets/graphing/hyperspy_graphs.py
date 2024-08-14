
#%%
%matplotlib qt
import hyperspy.api as hs
import numpy as np
import matplotlib.pyplot as plt

#%%
my_np_array = np.random.random((10, 20, 100))
s = hs.signals.Signal1D(my_np_array)
s

# %%
hs.preferences.gui()
# %%
