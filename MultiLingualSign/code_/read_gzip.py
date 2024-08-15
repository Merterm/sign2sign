import gzip 
import json
import ast
# a_file = gzip.open("../bobsl/5105777664862718183.tar.gz", "rb")
import tarfile
tar = tarfile.open("../bobsl/5105777664862718183.tar.gz")
tar.extractall()
tar.close()