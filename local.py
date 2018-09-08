import sys
import string
import math

filename = '/scratch/ds222-2017/assignment-1/DBPedia.full/full_train.txt'



class SpamDetector(object):
	def tokenize(self, text):
        text = self.clean(text).lower()
        return re.split('\W+', text)

	def get_word_counts(self, words):
        word_counts = {}
        for word in words:
                word_counts[word] = word_counts.get(word, 0.0) + 1.0
        return word_counts

	def fit(self, X, Y):
        self.log_class_priors = {}
        self.word_counts = {}
        self.vocab = set()


        n = len(x)
        for y in Y:
                for cls in y:
                        if cls not in self.log_class_priors:
                                self.log_class_priors[cls] = 0.0
                        self.log_class_priors[cls] += 1.0

        for cls, freq in self.log_class_priors.items():
                self.log_class_priors[cls] = freq/len(self.log_class_priors)




if __name__ == '__main__':
	with open(filename, encoding = 'latin-1') as f:
        temp = f.read()

	temp = temp.split("\n")
	print(temp[0])

	targets = []
	data = []

	print(temp[0].split("\t"))
	translator = str.maketrans("", "", string.punctuation)
	for t in temp:
	        line = t.split("\t")
	        if len(line) < 2:
	                continue
	        a = line[0]
	        b = line[1]
	        c = a.split(',')
	        d = []
	        for z in c:
	                d.append(z.strip())
	        targets.append(d)
	        data.append(b.translate(translator))
	print(len(targets))
	print(len(data))

	MNB = SpamDetector()
	MNB.fit(data, targets)