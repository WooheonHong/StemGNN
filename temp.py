import pandas as pd
import os
import numpy as np

data_file = os.path.join("dataset", "metr-la" + ".h5")

df = pd.read_hdf("dataset/metr-la.h5")
print(df)

# %%
np.array([[-0.00870192], [-0.01294216], [0.02450573], [0.01581013]])

# %%
import torch

residual = torch.rand(64, 32, 227, 12)
x = torch.rand(64, 32, 227, 10)
print(residual.size(3))
x = x + residual[:, :, :, -x.size(3) :]
x.shape
