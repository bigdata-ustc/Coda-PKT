import pickle
import os
data_path = "../data/PDKT"
with open(os.path.join(data_path, "all_results.pkl"), "rb") as f:
    all_results = pickle.load(f)

print("all_results: ", all_results)