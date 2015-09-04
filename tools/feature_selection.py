import math

class WordStatis():
	def __init__(self):
		self.tf_idf = {}
		self.class_word_df = {} #document frequency of word in every class
		self.chi_square = {}

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
		"""chi-square of words"""
		for doc,label in docs_with_label:
			#chi-square test doesn't consider term frequency
			unique_words = set(doc)
			#estimate the number of documents with the class 
			self.class_word_df.setdefault(label, {})
			self.class_word_df[label].setdefault('_class_count_', 0)
			self.class_word_df[label]['_class_count_'] += 1
			for word in unique_words:
				self.class_word_df[label].setdefault(word, 0.0)
				self.class_word_df[label][word] += 1.0

				self.chi_square.setdefault(word, {})
				self.chi_square[word].setdefault(label, 0.0)

	def calc_x2_of_class(self, class_label):
		"""calc chi-square of words"""
		if class_label not in self.class_word_df:
			return None

		other_class = filter(lambda y:y != class_label, self.class_word_df)
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





						