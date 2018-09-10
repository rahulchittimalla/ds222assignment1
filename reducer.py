from operator import itemgetter
import sys

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

current_word = None
current_count = 0
word = None

freq = {}

for line in sys.stdin:
	line = line.strip()
	word, count = line.split(',', 1)
	
	try:
		count = int(count)
	except ValueError:
		continue
	if current_word == word:
		current_count += 1
	else:
		if current_word:
			freq[current_word] = current_count
			print('{}\t{}'.format(current_word, current_count))
		current_count = count
		current_word = word
if current_word == word:
	freq[current_word] = current_count
	print('{}\t{}'.format(current_word, current_count))



start = time.time()
MNB = SpamDetector()
MNB.process_dict(freq)
X, Y = read_data(test_filename)
# print(len(X))
pred = MNB.predict(X, Y)

accuracy = sum(i for i in pred) / float(len(pred))
print("{0:.4f}".format(accuracy))
end = time.time()
print('testing time: {}'.format((end - start)/60))

class SpamDetector(object):

	def process_dict(self, freq):
		self.word_counts = freq
		self.clses  = []
		self.log_class_priors = {}
		for word, count in freq.items():
			w = word.split('and')
			if w[0].split('=')[1] not in clases:
				clases.append(w)
		for clas in clses:
			self.log_class_priors[clas] = 0.0

	def get_word_counts(self, words):
		word_counts = {}
		for word in words:
			word_counts[word] = word_counts.get(word, 0.0) + 1.0
		return word_counts

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


