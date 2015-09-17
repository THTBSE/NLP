from __future__ import unicode_literals
import pdb,sys,jieba

class TextRank():
	def __init__(self, window, epsilon=0.0001, damping_factor=0.85):
		self.window = window
		self.d = damping_factor
		self.epsilon = epsilon

	def kw_rank(self, doc, topK=5, combined=True):
		graph = {}
		wordindex = {}  #be used for conbine continous word
		#build a undirected and unweighted graph
		N = self.window - 1
		for i,word in enumerate(doc):
			wordindex.setdefault(word,i)
			graph.setdefault(word, {'adj':set(), 'score':1.0, 'error':10000})
			left = max(0, i-N)
			for adj in doc[left:i]:
				graph[adj]['adj'].add(word)
				graph[word]['adj'].add(adj)

		#iterate untial all vertices were converged
		converged_v = set()
		vertices_count = len(graph)
		while len(converged_v) < vertices_count:
			for v in graph:
				if graph[v]['error'] < self.epsilon:
					converged_v.add(v)
					continue
				score_old = graph[v]['score']
				graph[v]['score'] = (1 - self.d) + self.d * sum([graph[vj]['score']/len(graph[vj]['adj']) for vj in graph[v]['adj']])
				graph[v]['error'] = abs(graph[v]['score'] - score_old)

		#rank the vertices
		keywords = sorted(graph, key=lambda v:graph[v]['score'], reverse=True)
		if not combined:
			return keywords[:topK]

		# kw_score = []
		# for kw in keywords[:topK]:
		# 	kw_score.append((kw, graph[kw]['score']))
 		
 		#combine the continous keyword
 		order_words = [wordindex[w] for w in keywords[:topK] if w in wordindex]
 		order_words.sort()
 		cb_words = []

 		cb_word = ''
 		last_i = len(order_words) - 1
 		for i,index in enumerate(order_words):
 			next = min(i+1, last_i)
 			next_index = order_words[next]
 			if next_index - index == 1:
 				if not cb_word:
 					cb_word = doc[index] + doc[next_index]
 				else:
 					cb_word += doc[next_index]
 			else:
 				if cb_word:
 					cb_words.append(cb_word)
 					cb_word = ''
 				else:
 					cb_words.append(doc[index])

		return cb_words

if __name__ == '__main__':
	stop_words = set()
	for line in open('stopwords.txt'):
		line = line.rstrip()
		stop_words.add(line)

	test_doc = ''
	for line in open('test_doc.txt'):
		line = line.rstrip()
		test_doc = line

	seg_list = jieba.cut(test_doc)
	words = []

	for word in seg_list:
		if (word not in stop_words) and len(word) > 1:
			words.append(word)

	textrank = TextRank(3)

	num_keywords = int((1.0/3) * len(words))
	kw = textrank.kw_rank(words,num_keywords,True)

	for w in kw:
		print w