from os import listdir
from os.path import isfile, join
import os

cwd = os.getcwd()

path = os.path.join(cwd,"cleaneddata")

onlyfiles = [f for f in listdir(path) if isfile(join(path, f)) if f.endswith(".dat")]

onlyfiles = [f.split('.dat', 1)[0] for f in onlyfiles]


dictlist = [{'label': f, 'value': f} for f in onlyfiles]

print(dictlist)