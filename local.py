import sys

filename = "/scratch/ds222-2017/assignment-1/DBPedia.verysmall/"


data = []

with open(filename, encoding = 'latin-1') as f:
	data = f.read()

print(data[0])
