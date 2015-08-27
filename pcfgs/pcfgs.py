import sys, json
from collections import defaultdict

class PCFGs():
	def __init__(self):
		self.unary = defaultdict(int)
		self.binary = defaultdict(int)
		self.nonterm = defaultdict(int)

		self.rule_proba = {}
		self.temp_sentence = None


	def print_count(self):
		for term, count in self.nonterm.iteritems():
			print count, "NONTERMINAL", term

		for (term, word), count in self.unary.iteritems():
			print count, "UNARYRULE", term, word

		for (term, r1, r2), count in self.binary.iteritems():
			print count, "BINARYRULE", term, r1, r2

	def count(self, tree):
		if isinstance(tree, basestring):
			return

		if len(tree) == 3:
			self.nonterm[tree[0]] += 1
			self.binary[(tree[0],tree[1][0],tree[2][0])] += 1
			self.count(tree[1])
			self.count(tree[2])
		elif len(tree) == 2:
			self.nonterm[tree[0]] += 1
			self.unary[(tree[0], tree[1])] += 1

	def map_word(self,word):
		if word.isdigit():
			return '_Numeric_'
		elif word.isupper():
			return '_All_Capitals_'
		elif word[0].isupper():
			return '_First_Capital_'
		else:
			return '_RARE_'

	def map_lowfreq_word(self):
		lowfreq = filter(lambda key:self.unary[key] < 5, self.unary)
		for term_word in lowfreq:
			rare = self.map_word(term_word[1])
			key = (term_word[0], rare)
			count = self.unary[term_word]
			self.unary[key] += count
			self.nonterm[term_word[0]] += count 

	def load_file(self,parse_file):
		for l in open(parse_file, 'r'):
			t = json.loads(l)
			self.count(t)
		#self.print_count()

	def get_proba(self):
		#calculate probability of unary rule
		for (term, word), count in self.unary.iteritems():
			self.rule_proba.setdefault(word, {})
			self.rule_proba[word][term] = float(count) / self.nonterm[term]

		#calculate probability of binary rule
		for (term, r1, r2), count in self.binary.iteritems():
			self.rule_proba.setdefault((r1,r2), {})
			self.rule_proba[(r1,r2)][term] = float(count) / self.nonterm[term]

	def train(self, parse_file):
		self.load_file(parse_file)
		self.map_lowfreq_word()
		self.get_proba()

	def build_tree(self, tree, bp, i, j, s):
		if i != j:
			tree.append(s)
			tree.extend([[], []])
			k, r1, r2 = bp[i][j][s]
			self.build_tree(tree[1],bp,i,k,r1)
			self.build_tree(tree[2],bp,k+1,j,r2)
		else:
			tree.append(s)
			tree.append(self.temp_sentence[i])


	def probabilistic_cky(self, sentence):
		words_cnt = len(sentence)
		back_trace = [[{} for i in range(words_cnt)] for j in range(words_cnt)]
		table = [[defaultdict(float) for i in range(words_cnt)] for j in range(words_cnt)]
		for j,word in enumerate(sentence):
			if word not in self.rule_proba:
				word = self.map_word(word)
			for term in self.rule_proba[word]:
				table[j][j][term] = self.rule_proba[word][term]
			
			prev = range(j)
			prev.reverse()
			for i in prev:
				for k in range(i, j):
					for r1 in table[i][k]:
						for r2 in table[k+1][j]:
							binary = (r1,r2)
							if binary not in self.rule_proba:
								continue
							for term in self.rule_proba[binary]:
								prob = self.rule_proba[binary][term] * table[i][k][r1] * table[k+1][j][r2]
								if table[i][j][term] < prob:
									table[i][j][term] = prob

									back_trace[i][j][term] = (k, r1, r2)

		tree = []
		first,last = 0,words_cnt-1
		try:
			s = max(table[0][-1], key=lambda x:table[0][-1][x])
			self.temp_sentence = sentence
			self.build_tree(tree, back_trace, first, last, s)
		except ValueError,e:
			print sentence

		return tree 

	def predicts(self,predict_file):
		trees = []
		i = 0
		for line in open(predict_file,'r'):
			line = line.rstrip()
			sentence = line.split(' ')
			tree = self.probabilistic_cky(sentence)
			js_tree = json.dumps(tree)
			trees.append(js_tree)
			print 'No.{0} is Done'.format(i)
			i += 1
			sys.stdout.flush()
		return trees


if __name__ == '__main__':
	if len(sys.argv) < 3:
		sys.exit(1)

	pcfgs = PCFGs()
	pcfgs.train(sys.argv[1])
	trees = pcfgs.predicts(sys.argv[2])

	output = 'parse_dev1.dat'
	f = open(output, 'w')
	for tree in trees:
		f.write('{0}\n'.format(tree))
	f.close()
	


