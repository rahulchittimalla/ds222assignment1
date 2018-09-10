import pydoop.hdfs as hdfs
import string

train_filename = '/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt'
dev_filename = '/scratch/ds222-2017/assignment-1/DBPedia.full/full_devel.txt'
test_filename = '/scratch/ds222-2017/assignment-1/DBPedia.full/full_test.txt'


def read_data(filename):
	data = []
	targets = []
	cat = subprocess.Popen(["hadoop", "fs", "-cat", filename], stdout=subprocess.PIPE)
	for line in cat.stdout:
		x = line.split('\t')
		data = x[1]
		targets = x[0]

	return data, 
def clean(s):
	translator = str.maketrans("", "", string.punctuation)
	return s.translate(translator)

def tokenize(text):
	text = clean(text).lower()
	return re.split('\W+', text) 

if __name__ == '__main__':
	X, Y = read_data(train_filename)

	for x in X:
		x = tokenize(x)
		for y in Y:
			print('Y={}, 1'.format(y))
			print('Y=ANY, 1')
		for w in x:
			for y in Y:
				print('Y={} and W={}, 1'.format(y,w))
				print('Y={} and W=ANY, 1'.format(y))
				


