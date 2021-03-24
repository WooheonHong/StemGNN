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
print(x.size(3))
x = x + residual[:, :, :, -x.size(3) :]
x.shape

# %%
import torch

x = torch.arange(4)
print(x)
ffted = torch.fft.fft(x)
real = ffted.real
imag = ffted.imag
print(ffted.real)
print(ffted.imag)
time_step_as_inner = torch.stack((real, imag), -1)
print("time_step_as_inner: ", time_step_as_inner.shape)
time_step_as_inner = torch.view_as_complex(time_step_as_inner)
iffted = torch.fft.ifft(time_step_as_inner)
# iffted = torch.fft.irfft(time_step_as_inner)  # (((12/6) + 1 - 2 ) - 1) * 2-> 8
# iffted = torch.istft(time_step_as_inner, 4)
print("iffted shape: ", iffted.shape)
print(iffted.real)
