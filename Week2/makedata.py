import numpy as np
import pandas as pd

def data_table():
  normtable = pd.read_csv("/Users/neymikajain/Desktop/SURF22/Week2/data_normalized.csv", header=None)
  xi = np.array([normtable.iloc[0, 0:48]])
  yi = np.array([normtable.iloc[0, 48]])
  for i in range(1, 200):
      xi = np.append(xi, [normtable.iloc[i, 0:48]], axis=0)
      yi = np.append(yi, [normtable.iloc[i, 48]], axis=0)
  return xi, yi

def main():
    return data_table()

if __name__ == "__main__":
    data_table
