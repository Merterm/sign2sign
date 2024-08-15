import json 
import pickle
from xmlrpc.server import list_public_methods 

# a = json.load(open("../bobsl/spottings/dict_spottings.json"))
# print(list(a['public_test']['image']['probs']))
# print(list(a))

g = pickle.load(open("../bobsl/spottings/bobsl_mouthing_c2281_verified_mouthing_9263_dict_15782.pkl", "rb"))
# json.dump(g, open("../bobsl/spottings/verified_dicst.json", "w"))
# print(len(g['videos']['name']))
print(list(g['videos']['videos']))