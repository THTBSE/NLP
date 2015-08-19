import sys
from collections import defaultdict

#包装一个读取一个词与标注的迭代器
def corpus_iterator(corpus_file):
	l = corpus_file.readline()
	while l:
		line = l.strip()
		if line:
			fields = line.split(' ')
			tag = fields[-1]
			word = ' '.join(fields[:-1])
			yield word, tag
		else:
			yield (None, None)
		l = corpus_file.readline()

#包装一个返回一个句子的迭代器
def sentence_iterator(corpus_iter):
	sentence = []
	for l in corpus_iter:
		if l == (None,None):
			if sentence:
				yield sentence
				sentence = []
			else:
				sys.stderr.write('Error:Empty files!')
				raise StopIteration
		else:
			sentence.append(l)
	if sentence:
		yield sentence

#一次返回ngram的迭代器
def ngram_iterator(sentence_iter, n):
	for sent in sentence_iter:
		words = (n-1) * [(None,'*')]
		words.extend(sent)
		words.append((None,'STOP'))
		ngrams = (tuple(words[i:i+n]) for i in xrange(len(words)-n+1))
		for ngram in ngrams:
			yield ngram

class Hmm():
	def __init__(self, n=3):
		self.n = n
		self.emission_counts = defaultdict(int)
		self.ngram_counts = [defaultdict(int) for i in range(self.n)]
		self.all_states = set() #所有标注状态
		self.emission_prob = {} #状态与词的输出概率
		self.state_trans_prob = {} #状态转移概率

	#把低频词映射为pseudo-word
	def map_lowfreq_word(self, word):
		#如果有至少一个数字
		n = filter(lambda x:x.isdigit(), word)
		if n:
			return '_Numeric_'
		#全是大写字母
		if word.isupper():
			return '_All_Capitals_'
		#末位大写
		if word[-1].isupper():
			return '_Last_Capital_'
		return '_Rare_'

	def map_all_lowfreq_words(self, r=5):
		lowfreq = set()
		rare_counts = defaultdict(int)
		for word_tag in self.emission_counts:
			count = self.emission_counts[word_tag]
			if count < r:
				pseudo_word = self.map_lowfreq_word(word_tag[0])
				rare_counts[(pseudo_word, word_tag[-1])] += count
				lowfreq.add(word_tag)

		for lf in lowfreq:
			self.emission_counts.pop(lf)
		for rare_tag in rare_counts:
			self.emission_counts[rare_tag] = rare_counts[rare_tag]

	def get_emission_prob(self):
		for word_tag in self.emission_counts:
			tag_count = self.ngram_counts[0][word_tag[-1:]]
			self.emission_prob.setdefault(word_tag[0],defaultdict(float))
			self.emission_prob[word_tag[0]][word_tag[-1]] = float(self.emission_counts[word_tag]) / tag_count

	def get_state_trans_prob(self):
		for ngram in self.ngram_counts[self.n - 1]:
			prev_gram = ngram[:-1]
			v = ngram[-1]
			self.state_trans_prob.setdefault(prev_gram, {})
			self.state_trans_prob[prev_gram][v] = float(self.ngram_counts[self.n-1][ngram]) / self.ngram_counts[self.n-2][prev_gram]

	def train(self, corpus_file):
		ngram_iter = ngram_iterator(sentence_iterator(corpus_iterator(corpus_file)), self.n)

		for ngram in ngram_iter:
			assert len(ngram) == self.n, "ngram in stream is %i, expected %i" % (len(ngram, self.n))

			tagsonly = tuple([tag for word, tag in ngram])
			for i in range(2,self.n+1):
				self.ngram_counts[i-1][tagsonly[-i:]] += 1

			if ngram[-1][0] is not None:
				self.ngram_counts[0][tagsonly[-1:]] += 1
				self.emission_counts[ngram[-1]] += 1
				self.all_states.add(tagsonly[-1])

			#如果倒数第二个还是None，说明这是句子的开头
			if ngram[-2][0] is None:
				self.ngram_counts[self.n - 2][tuple((self.n - 1) * ["*"])] += 1

		self.map_all_lowfreq_words()
		self.get_emission_prob()
		self.get_state_trans_prob()

	def trigram_viterbi(self, observation):
		pi = {('*','*'):1.0}
		path = [{} for i in range(len(observation))]
		for t,x in enumerate(observation):
			next_pi = {} 
			for v in self.all_states:
				maxState = {'p':-1.0, 'bigram':None}
				word, tag = x, v
				if word not in self.emission_prob:
					word = self.map_lowfreq_word(word)
				for bigram in pi:
					prob = pi[bigram] * self.state_trans_prob[bigram][v] * self.emission_prob[word][tag]
					if prob > maxState['p']:
						maxState['p'] = prob
						maxState['bigram'] = (bigram[-1], v)
				next_pi[maxState['bigram']] = maxState['p']
				#时刻t状态v的前一个概率最大状态
				path[t][v] = maxState['bigram'][0]
			pi = next_pi

		maxProb = -1.0
		last_state = None
		for bigram in pi:
			prob = pi[bigram] * self.state_trans_prob[bigram]['STOP']
			if prob > maxProb:
				maxProb = prob
				last_state = bigram[-1]

		#从最后一个状态点回溯出最佳路径
		seq = []
		#last_state = max(pi, key=lambda k:pi[k])[-1]
		seq.append(last_state)
		path.reverse()
		for bp in path[:-1]:
			prev_state = bp[last_state]
			seq.append(prev_state)
			last_state = prev_state
		seq.reverse()
		return seq

	def write_count(self, ngrams=[1,2,3]):
		for n in ngrams:
			for ngram in self.ngram_counts[n-1]:
				ngramstr = " ".join(ngram)
				print "%i %i-GRAM %s" % (self.ngram_counts[n-1][ngram], n, ngramstr) 

def predicts(input_file, output_file, hmm_model):
	f = open(input_file,'r')
	output = open(output_file,'w')
	sentence = []
	for line in f:
		line = line.strip()
		if line:
			sentence.append(line)
		else:
			seq = hmm_model.trigram_viterbi(sentence)
			for word, tag in zip(sentence, seq):
				output.write('{0} {1}\n'.format(word,tag))
			output.write('\n')
			sentence = []
	output.close()
	f.close()

if __name__ == '__main__':
	if len(sys.argv) != 3: 
		sys.exit(2)

	try:
		f = open(sys.argv[1], 'r')
	except IOError:
		sys.stderr.write("ERROR: Cannot read inputfile %s.\n" % arg)
		sys.exit(1)

	# Initialize a trigram hidden Markov model 
	model = Hmm(3)
	# Train the model
	model.train(f)
	f.close()
	predicts(sys.argv[2], 'gene.dev1', model)

