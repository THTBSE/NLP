from __future__ import unicode_literals
import re

MIN_FLOAT = -3.14e100

trans_P = None
emit_P = None

from prob_trans import P as trans_P
from prob_emit import P as emit_P

def strdecode(sentence):
    if not isinstance(sentence, unicode):
        try:
            sentence = sentence.decode('utf-8')
        except UnicodeDecodeError:
            sentence = sentence.decode('gbk','ignore')
    return sentence

def viterbi(obs, states, trans_p, emit_p):
    pi = {'*':0.0}
    obs_length = len(obs)
    path = [{} for i in range(obs_length)]
    for t,x in enumerate(obs):
        nextpi = {}
        for w in states:
            nextpi[w], path[t][w] = max([(trans_p[v].get(w, MIN_FLOAT)+emit_p[w].get(x, MIN_FLOAT)+pi[v], v) for v in pi])
        pi = nextpi

    last_state, prob = max(pi.iteritems(), key=lambda x:x[1])

    pos_list = ['B' for i in range(obs_length)]
    index = obs_length - 1
    while index >= 0:
        pos_list[index] = last_state
        last_state = path[index][last_state]
        index -= 1

    return (prob, pos_list)

def __cut(sentence):
    prob, pos_list = viterbi(sentence, 'BMES', trans_P, emit_P)
    beg = 0
    nexti = 0
    for i,x in enumerate(pos_list):
        if x == 'B':
            beg = i
        elif x == 'E':
            yield sentence[beg:i+1]
            nexti = i + 1
        elif x == 'S':
            yield sentence[i]
            nexti = i + 1
    if nexti < len(sentence):
        yield sentence[nexti:]

re_chi = re.compile('([\u4E00-\u9FA5]+)')
re_alnum = re.compile('(\d+.\d+|[a-zA-Z0-9]+)')

def chi_cut(sentence):
    sentence = strdecode(sentence)
    blocks = re_chi.split(sentence)
    for block in blocks:
        if re_chi.match(block):
            for word in __cut(block):
                yield word
        else:
            alnum = re_alnum.split(block)
            for x in alnum:
                if x:
                    yield x
