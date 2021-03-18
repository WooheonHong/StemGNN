import pandas as pd
import os

data_file = os.path.join("dataset", "metr-la" + ".h5")

df = pd.read_hdf("dataset/metr-la.h5")
print(df)
