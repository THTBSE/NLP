import math
from collections import defaultdict

class WordStatis():
	def __init__(self):
		self.tf_idf = {} #using for set weight for feature words 
		self.class_word_df = {} #document frequency of word in every class
		self.chi_square = {}
		self.information_gain = {}
		self._all_words = set()

	def tf(self, docs):
		"""term frequency of words"""
		total = 0
		for doc in docs:
			for word in doc:
				self.tf_idf.setdefault(word, {})
				self.tf_idf[word].setdefault('tf',0)
				self.tf_idf[word]['tf'] += 1
				total += 1
		total = float(total)

		for word in self.tf_idf:
			self.tf_idf[word]['tf'] /= total

	def idf(self, docs):
		"""inverse documents frequency of words"""
		doc_num = 0
		for doc in docs:
			unique_words = set(doc)
			for word in unique_words:
				self.tf_idf.setdefault(word, {})
				self.tf_idf[word].setdefault('idf',0)
				self.tf_idf[word]['idf'] += 1
			doc_num += 1

		doc_num = float(doc_num)
		for word in self.tf_idf:
			self.tf_idf[word]['idf'] = math.log(doc_num / (self.tf_idf[word]['idf']+1))

	def calc_tf_idf(self):
		"""tf_idf of words"""
		if not self.tf_idf:
			print 'Please calculate tf and idf first'

		for word in self.tf_idf:
			self.tf_idf[word]['tf_idf'] = self.tf_idf[word]['tf'] * self.tf_idf[word]['idf']

	def calc_class_word_df(self, docs_with_label):
		"""calc every word's document frequency in every class"""
		for doc,label in docs_with_label:
			#chi-square test doesn't consider term frequency
			unique_words = set(doc)
			#estimate the number of documents with the class 
			self.class_word_df.setdefault(label, defaultdict(float))
			self.class_word_df[label]['_class_count_'] += 1.0
			for word in unique_words:
				self.class_word_df[label][word] += 1.0
				self._all_words.add(word)

	def calc_x2_of_class(self, class_label):
		"""calc chi-square of words"""
		if class_label not in self.class_word_df:
			return None

		all_classes = self.class_word_df.keys()
		self.chi_square = {w:{c:0.0 for c in all_classes} for w in self._all_words}
		other_class = filter(lambda y:y != class_label, all_classes)

		for word in self.chi_square:
			A = B = C = D = 0.0
			if word in self.class_word_df[class_label]:
				A = self.class_word_df[class_label][word]
			C = self.class_word_df[class_label]['_class_count_'] - A

			other_count = 0
			for cls in other_class:
				if word in self.class_word_df[cls]:
					B += self.class_word_df[cls][word]
				other_count += self.class_word_df[cls]['_class_count_']
			D = other_count - B

			N = A + B + C + D
			self.chi_square[word][class_label] = (N*((A*D - B*C)**2)) / ((A+C)*(A+B)*(B+D)*(C+D))

	def calc_information_gain(self):
		"""calc IG for features, the class_word_df should be calc first"""
		if not self.class_word_df:
			return None

		all_classes = self.class_word_df.keys()
		prob_class = {k:0.0 for k in all_classes}
		all_docs = 0

		#estimate probability of each class
		for cls in all_classes:
			clsCount = self.class_word_df[cls]['_class_count_']
			prob_class[cls] += clsCount
			all_docs += clsCount

		all_docs = float(all_docs)
		for cls in all_classes:
			prob_class[cls] /= all_docs

		#entropy of original dataset
		entropy = 0.0
		for cls in all_classes:
			entropy -= (prob_class[cls] * math.log(prob_class[cls])) 

		for word in self._all_words:
			#p(w) and p(_w)
			prob_word = 0.0
			num_word = 0.0
			for cls in all_classes:
				num_word += self.class_word_df[cls][word]
			prob_word = num_word / all_docs
			num_non_word = all_docs - num_word
			prob_non_word = 1 - prob_word

			#p(c|w) and p(c|_w)
			prob_word_class = {k:0.0 for k in all_classes}
			prob_non_word_class = {k:0.0 for k in all_classes}

			for cls in all_classes:
				prob_word_class[cls] = self.class_word_df[cls][word] / num_word
				prob_non_word_class[cls] = (self.class_word_df[cls]['_class_count_'] - self.class_word_df[cls][word]) / num_non_word


			entropy_word = entropy_non_word = 0.0
			for cls in all_classes:
				if prob_word_class[cls] > 0:
					entropy_word -= (prob_word_class[cls] * math.log(prob_word_class[cls]))
				if prob_non_word_class[cls] > 0:
					entropy_non_word -= (prob_non_word_class[cls] * math.log(prob_non_word_class[cls]))

			#IG = H(X) - H(C|X)
			self.information_gain[word] = entropy - (entropy_word + entropy_non_word)
						