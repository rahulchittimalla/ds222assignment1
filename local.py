import sys
import string
import math
import re
import operator
import time

train_filename = '/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt'
dev_filename = '/scratch/ds222-2017/assignment-1/DBPedia.full/full_devel.txt'
test_filename = '/scratch/ds222-2017/assignment-1/DBPedia.full/full_test.txt'




class SpamDetector(object):
	def clean(self, s):
		translator = str.maketrans("", "", string.punctuation)
		return s.translate(translator)

	def tokenize(self, text):
  		text = self.clean(text).lower()
  		return re.split('\W+', text)      

	def get_word_counts(self, words):
		word_counts = {}
		for word in words:
			word_counts[word] = word_counts.get(word, 0.0) + 1.0
		return word_counts

	def fit(self, X, Y):
		print("fitting the data")
		self.clses = []
		self.log_class_priors = {}
		self. word_counts = {}
		self.vocab = set()

		n = len(X)
		for y in Y:
			for cls in y:
				if cls not in self.log_class_priors:
					self.log_class_priors[cls] = 0.0
				self.log_class_priors[cls] += 1.0
		for cls, freq in self.log_class_priors.items():
			self.log_class_priors[cls] = math.log(freq/n)

		for cls,_ in self.log_class_priors.items():
			self.word_counts[cls] = {}
			self.clses.append(cls)

		for x, y in zip(X, Y):
			counts = self.get_word_counts(self.tokenize(x))
			for word, count in counts.items():
				if word not in self.vocab:
					self.vocab.add(word)
				for cls in y:
					if word not in self.word_counts[cls]:
						self.word_counts[cls][word] = 0.0
					self.word_counts[cls][word] += count

	def calculate_prob(self):
		self.probs = {}
		for clas in self.clses:
			self.probs[clas] = {}
		for clas in self.clses:
			total_sum = sum(self.word_counts[clas].values())
			for word in self.vocab:
				self.probs[clas][word] = math.log((self.word_counts[clas].get(word, 0.0)+1.0)/float(total_sum + len(self.vocab)))

	def predict(self, X, Y):
		print('calculating probabilities')
		self.calculate_prob()
		print('prediction starts')
		result = []
		for x,y  in zip(X, Y):
			counts = self.get_word_counts(self.tokenize(x))
			scores = {}
			for clas in self.clses:
				scores[clas] = 0.0
			for word, _ in counts.items():
				if word not in self.vocab: continue 

				# add laplace smoothing
				for clas in self.clses:
					# scores[clas] += math.log((self.word_counts[clas].get(word, 0.0)+1.0)/float(sum(self.word_counts[clas].values()) + len(self.vocab)))
					scores[clas] += self.probs[clas][word]
				
			for clas, score in scores.items():
				scores[clas] += self.log_class_priors[clas]

			max_clas = max(scores.items(), key=operator.itemgetter(1))[0]
			# print(max_clas)

			flag = False

			if max_clas in y:
				result.append(1)
				# print('true')
			else:
				result.append(0)
				# print('false')
			# for clas  in y:
			# 	print(clas, max_clas)
			# 	if clas == max_clas:
			# 		result.append(1)
			# 		flag = True
			# 		break

			# if flag == False:
			# 	result.append(0)

		return result


def read_data(filename):
	with open(filename, encoding = 'latin-1') as f:
		temp = f.read()

	temp = temp.split('\n')
	print(temp[0])

	targets = []
	data = []

	print(temp[0].split('\t'))

	for t in temp:
		line = t.split('\t')
		if len(line) < 2:
			continue
		a = line[0]
		b = line[1]
		c = a.split(',')
		d = []
		for z in c:
			d.append(z.strip())
		targets.append(d)
		data.append(b)
	print(len(targets))
	print(len(data))
	return data, targets


if __name__ == '__main__':
	
	start = time.time()
	data, targets = read_data(train_filename)
	MNB = SpamDetector()
	MNB.fit(data, targets)
	end = time.time()
	print('training time: {}'.format((end-start)/60))
	X, Y = read_data(test_filename)
	# print(len(X))
	start = time.time()
	pred = MNB.predict(X, Y)

	accuracy = sum(i for i in pred) / float(len(pred))
	print("{0:.4f}".format(accuracy))
	end = time.time()
	print('testing time: {}'.format((end - start)/60))

