import pandas as pd
import pickle 

with open('testing_set.pkl', 'rb') as f:
    data = pickle.load(f)

print(data)