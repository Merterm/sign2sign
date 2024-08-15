import os
import pickle

for fname in os.listdir("output"):
    if "test_results.pkl" in fname:
        print(fname)
        g = pickle.load(open("output/" + fname,"rb" ))
        print(g['validate_scores'])
        print()
