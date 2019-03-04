# Lexical-Substitution-Task-using-WordNet-and-Word2Vec-Word-Embeddings
Using WordNet and pretrained Word2Vec word Embeddings to solve the lexical substitution task (that was first proposed as a shared task at SemEval 2007 Task 10). 
In this task, the goal is to find lexical substitutes for individual target words in context. For example, given the following sentence:

"Anyway , my pants are getting tighter every day ." The goal is to propose an alternative word for tight, such that the meaning of the sentence is preserved. Such a substitute could be constricting, small or uncomfortable.

In the sentence

"If your money is tight don't cut corners ." the substitute small would not fit, and instead possible substitutes include scarce, sparse, constricted. You will implement a number of basic approaches to this problem. You also have the option to improve your solution and e

# Candidate Synonyms from WordNet 
The function get_candidates(lemma, pos) takes a lemma and part of speech ('a','n','v','r') as parameters and returns a set of possible substitutes. To do this, we look up the lemma and part of speech in WordNet and retrieve all synsets that the lemma appears in. Then obtain all lemmas that appear in any of these synsets. For example,
```
>>> get_candidates('slow','a')
{'deadening', 'tiresome', 'sluggish', 'dense', 'tedious', 'irksome', 'boring', 'wearisome', 'obtuse', 'dim', 'dumb', 'dull', 'ho-hum'}
```

# WordNet Frequency Baseline 
The function wn_frequency_predictor(context) takes a context object as input and predicts the possible synonym with the highest total occurence frequency (according to WordNet). We have to sum up the occurence counts for all senses of the word if the word and the target appear together in multiple synsets. The get_candidates method can be used for this or we can just duplicate the code for finding candidate synonyms (this is possibly more convenient). Using this simple baseline should give you about 10% precision and recall. 

# Simple Lesk Algorithm
The function wn_simple_lesk_predictor(context) uses Word Sense Disambiguation (WSD) to select a synset for the target word. It returns the most frequent synonym from that synset as a substitute. To perform WSD, we implement the simple Lesk algorithm. Look at all possible synsets that the target word apperas in. Compute the overlap between the definition of the synset and the context of the target word. We remove stopwords (function words that don't tell you anything about a word's semantics). We also load the list of English stopwords in NLTK like this:
```
stop_words = stopwords.words('english')
```
The main problem with the Lesk algorithm is that the definition and the context do not provide enough text to get any overlap in most cases. We therefore add the following to the definition:

* All examples for the synset.
* The definition and all examples for all hypernyms of the synset.
Even with these extensions, the Lesk algorithm will often not produce any overlap. If this is the case (or if there is a tie), you should select the most frequent synset (i.e. the Synset with which the target word forms the most frequent lexeme, according to WordNet). Then select the most frequent lexeme from that synset as the result. One sub-task that you need to solve is to tokenize and normalize the definitions and exmaples in WordNet. You could either look up various tokenization methods in NLTK or use the tokenize(s) method provided with the code. In my experiments, the simple lesk algorithm did not outperform the WordNet frequency baseline.

# Most Similar Synonym
We implement approaches based on Word2Vec embeddings. These will be implemented as methods in the class Word2VecSubst. The reason these are methods is that the Word2VecSubst instance can store the word2vec model as an instance variable. The constructor for the class Word2VecSubst already includes code to load the model. 

The method predict_nearest(context) first obtains a set of possible synonyms from WordNet, and then returns the synonym that is most similar to the target word, according to the Word2Vec embeddings. In my experiments, this approach worked slightly better than the WordNet Frequency baseline and resulted in a precision and recall of about 11%.

# Context and Word Embeddings 
In this part, we implement the method predict_nearest_with_context(context). One problem of the approach in part 4 is that it does not take the context into account. Like the model in part 2, it ignores word sense. There are many approaches to model context in distributional semantic models. For now, we will do something very simple. First create a single vector for the target word and its context by summing together the vectors for all words in the sentence, obtaining a single sentence vector. Then measure the similarity of the potential synonyms to this sentence vector. This works better if you remove stop-words and limit the context to +-5 words around the target word. In my experiments, this approach resulted in a precison and recall of about 12%.
