#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml

# suggested imports 
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import gensim
import numpy as np
import string
import math

# Participate in the 4705 lexical substitution competition (optional): NO
# Alias: [please invent some name]

def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos):
    # Part 1
    possible_synonyms = []
    for l in wn.lemmas(lemma, pos):
        sl = l.synset().lemmas()
        for s in sl:
            name = s.name()
            name = name.replace("_", " ")
            possible_synonyms.append(name)
    possible_synonyms = set(possible_synonyms)
    possible_synonyms.remove(lemma)
    return possible_synonyms

def smurf_predictor(context):
    """
    Just suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context):
    canidates = list(get_candidates(context.lemma, context.pos))
    occurence_counts = {canidates[i]:0 for i in range(len(canidates))}

    #possible_synonyms = []
    for l in wn.lemmas(context.lemma, context.pos):
        sl = l.synset().lemmas()
        for s in sl:
            name = s.name()
            name = name.replace("_", " ")
            if name in occurence_counts.keys():
                occurence_counts[name] = occurence_counts[name] + 1
    for word, count in occurence_counts.items():
        if count == max(occurence_counts.values()):
            #print(word, count, )
            return word
    return None

def wn_simple_lesk_predictor(context):
    Lem = WordNetLemmatizer()
    definitions_for_synset = dict()

    lr_context = set().union(context.left_context, context.right_context)
    lr_context = list(lr_context.difference(set(stopwords.words("english"))))
    for i in range(len(lr_context)):
        lr_context[i] = Lem.lemmatize(lr_context[i])

    for s in wn.synsets(context.lemma):
        if not s.lemmas()[0].name() == context.lemma:
            definitions = []
            definitions.append(s.definition())
            for e in s.examples():
                definitions.append(e)
            for h in s.hypernyms():
                definitions.append(h.definition())
                for e in h.examples():
                    definitions.append(e)
            definitions_for_synset[s] = definitions
    overlap_count = {list(definitions_for_synset.keys())[i]:0 for i in range(len(definitions_for_synset.keys()))}
    for syn, definitions in definitions_for_synset.items():
        for d in definitions:
            w = d.split(" ")
            for w1 in w:
                w3 = w1.lower()
                w3 = Lem.lemmatize(w3)
                for w2 in lr_context:
                    if w3 == w2.lower():
                        overlap_count[syn] = overlap_count[syn] + 1
    max_frequency = []
    for syn, count in overlap_count.items():
        if count == max(overlap_count.values()):
            max_frequency.append((syn.lemmas()[0].count(), syn.lemmas()[0].name()))
    if not len(overlap_count.keys()) == 0:
        return max(max_frequency)[1]
    else:
        return context.lemma
    return None 

def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None


def lemmatize(sentence):
    Lem = WordNetLemmatizer()

    tagged_sentence = nltk.pos_tag(sentence)
    lemmatized_sentence = []
    for i in range(len(tagged_sentence)):
        tag = tagged_sentence[i][1]
        pos = penn_to_wn(tag)
        if pos and not tagged_sentence[i][0] in stopwords.words("english"):
            lemmatized_sentence.append(Lem.lemmatize(tagged_sentence[i][0], pos))
    return lemmatized_sentence


class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context):
        possible_synonyms = get_candidates(context.lemma, context.pos)
        max_similarity = 0
        best_synonym = None
        for s in possible_synonyms:
            if s in self.model.wv.vocab:
                similarity = self.model.similarity(context.lemma, s)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_synonym = s
        return best_synonym # replace for part 4

    def predict_nearest_with_context(self, context): 
        vector_sum = self.model.wv[context.lemma]
        # sum together all the vectors for all words in the sentence
        lr_context = set().union(context.left_context[-5:], context.right_context[0:5])
        lr_context = list(lr_context.difference(set(stopwords.words("english"))))

        for c in lr_context:
            if c in self.model.wv.vocab:
                vector_sum = vector_sum + self.model.wv[c]

        possible_synonyms = get_candidates(context.lemma, context.pos)
        max_similarity = 0
        best_synonym = None
        for s in possible_synonyms:
            if s in self.model.wv.vocab:
                similarity = self.cos(self.model.wv[s], vector_sum)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_synonym = s
        if max_similarity < 0.3:
            return self.predict_nearest(context)
        return best_synonym

    def improved_predict_nearest_with_context(self, context): 
        vector_sum = self.model.wv[context.lemma]
        # sum together all the vectors for all words in the sentence
        s1 = tokenize(" ".join(context.left_context) + context.word_form + " ".join(context.right_context))
        s2 = lemmatize(s1)
        index = 0
        for i in range(len(s2)):
            if s2[i] == context.lemma:
                index = i
        sentence = []
        for i in range(len(s2)):
            if i > index-5 and i < index+5:
                sentence.append(s2[i])
        lr_context = []
        for s in sentence:
            if s not in stopwords.words("english"):
                lr_context.append(s)
        for c in lr_context:
            if c in self.model.wv.vocab:
                vector_sum = vector_sum + self.model.wv[c]

        possible_synonyms = get_candidates(context.lemma, context.pos)
        max_similarity = 0
        best_synonym = None
        for s in possible_synonyms:
            if s in self.model.wv.vocab:
                similarity = self.cos(self.model.wv[s], vector_sum)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_synonym = s
        #if max_similarity < 0.3:
        #    return self.predict_nearest(context)
        return best_synonym

    def cos(self, v1, v2):
        return np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))



if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6).
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
    predictor = Word2VecSubst(W2VMODEL_FILENAME)

    for context in read_lexsub_xml(sys.argv[1]):
        #prediction = wn_frequency_predictor(context)
        #prediction = wn_simple_lesk_predictor(context)
        #prediction = predictor.predict_nearest(context)
        #prediction = predictor.predict_nearest_with_context(context)
        prediction = predictor.improved_predict_nearest_with_context(context)
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))




  
