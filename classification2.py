import csv											# used for import csv tweets
import random										# used for train/test set creation
import nltk											# used for tokenization
import os											# used for calling Frog in new terminal
import operator
import time											# used for timer
from pynlpl.clients.frogclient import FrogClient	# used for Frog client
import subprocess									# used for calling Frog in new terminal
import signal										# used for calling Frog in new terminal
import math	 
import pickle										# write and load list to file
import nltk											# used for stemmer
from preprocessing import Preprocessing
from bagofwords import BagOfWords
from sklearn import svm
from sklearn.datasets import load_svmlight_file
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


import helpers
from start_svm import Start_SVM


class Main(object):
	"""
	Class for blablabla
	"""
	# Input file reader
	DELIMITER = "\t"
	data = csv.reader(open("2000test_annotated_v2.csv", 'rU'), delimiter=DELIMITER)

	# Dictionary for class, classify unkown into non-activity
	class_dict = {"Y": 0, "N": 1, "U": 0}

	# Distribution of trainingset, testset, validationset
	distribution = (0.7, 0.2, 0.1)

	testset = []
	trainset = []
	tweets = {}
	tweet_class = {}

	# Processed arrays
	stemmed_tweets_array = []
	tokenized_tweets_array = []
	pos_tweets_array = []
	lemmatized_tweets_array = []

	# Debug files
	DEBUG_SETS = "debug_sets.txt"
	
	# File for classification results
	RESULTFILE = 'results_classification_17juni.csv'

	# Arrays for classification
	train_tweetclasses = []
	train_vectors= []
	test_tweetclasses = []
	test_vectors = []

	# Cross validation folds
	CROSS_VALIDATION = 5

	# Scaler object
	scaler = None

	def __init__(self, testingmode, mode):
		""" Initialize tweets, preprocess and create train/test sets"""
		self.mode = mode
  		#self.debug = "--debug" in mode
  		#self.dump = "--write" in mode
		self.debug = True
		self.dump = False
		self.testingmode = testingmode

		self.initialize()
		self.preprocess_tweets()
		if ( self.testingmode ):
			self.create_sets()
			self.count_classes()

	def initialize(self):
		""" Initializes tweet and class sets """
		print "** Initialize.."
		for i, row in enumerate(self.data):
			if(i == 0):		# Ignore header
				pass
			else:
				# Get tweet and class of tweet 
				if (self.class_dict.get(row[5].upper()) is not None):
					self.tweets[i-1] = row[3]
					if ( self.testingmode ):
						self.tweet_class[i-1] = self.class_dict.get(row[5].upper())

	def start_svm_classification(self, mode, ngrambow, minborder, maxborder, nr):
		""" Runs classification learning"""
		# Create BOW
		array = self.get_preprocessed_array(mode)
		negbow, posbow = self.collect_bow(array, ngrambow, minborder, maxborder, nr/2)
		totalbow = dict(posbow.items() + negbow.items())
		self.scaler = None


		"""
		# Create train & test data (classes, vectors)
		self.train_tweetclasses, self.train_vectors = self.svm_create_traintestdata(array, posbow, negbow, self.trainset, mode)
		self.test_tweetclasses, self.test_vectors = self.svm_create_traintestdata(array, posbow, negbow, self.testset, mode)

		# Run SVM
		results = self.run_svm(np.array(self.train_vectors), np.array(self.train_tweetclasses), np.array(self.test_vectors), np.array(self.test_tweetclasses), self.CROSS_VALIDATION)
		"""
		tuplebows = negbow, posbow 
		svmObject = Start_SVM(array, self.tweet_class, self.trainset, self.testset, True, tuplebows, 5)

		results = svmObject.start_svm_testing(mode, minborder, maxborder, nr)
		return results

	def run_svm(self, X_train, y_train, X_test, y_test, k):
		""" Run SVM classifier. Configure parameters for SVM using grid search, then fit with cross valiation.
		Then predict test set. Return best parameters and scores.
		"""
		clf = svm.SVC()

		# Parameter grid
		param_grid = [
		 {'C': np.logspace(1,5,5), 'gamma': np.logspace(-3,0,5), 'kernel': ['rbf']}
		]

		score_func = metrics.f1_score
		clf = GridSearchCV(SVC(), param_grid, score_func=score_func,  n_jobs=-1 )

		###print "** Fitting SVM classifier.."
		clf.fit(X_train, y_train, cv=k)

		# Get best parameters
		dict_param = clf.best_params_
		gamma1 = dict_param['gamma']
		c = dict_param['C']

		# Get scores
		###print "** Run SVM classifier.."
		y_true, y_pred = y_test, clf.predict(X_test)
		tuples = precision_recall_fscore_support(y_true, y_pred)

		return (tuples, gamma1, c)

	def nb_create_traintestdata(self, array, ngram, posbow, negbow, indexset,mode, **kwargs):

		allwords = kwargs.get('allwords', False)
		tweets = []
		classes = []


		# Select BOW type
		if ('posneg' in mode):
			bow = dict(posbow.items() + negbow.items())
		if ('pos1' in mode):
			bow =  posbow
		if ('neg1' in mode):
			bow = negbow

		# Use all words in tweets
		if ( allwords ):
			for index in indexset:
				tweets.append(' '.join(array[index]))
				classes.append(self.tweet_class[index])



		
		# Use words occuring in BOW of tweet
		else:
			for index in indexset:
				tweet = array[index]
				bowtweet = self.get_bowtweet(tweet, bow, ngram)
				tweets.append(bowtweet)
				classes.append(self.tweet_class[index])
		
		return (classes, tweets)


	def get_bowtweet(self, tweet, bow, ngram_array):
		""" Get modified tweet with only words in bow"""
		tuple_array = self.splitbow(bow, ngram_array)
		
		listtweets = []
		# Set default to False
		for i in range(0, len(ngram_array)):
			listtweet = [False]*len(tweet)
			listtweets.append(listtweet)
		
		# Get array of Booleans for words occuring in BOW
		for index_n, ngram in enumerate(ngram_array):
			for index_t, word in enumerate(tweet):
				wordarray = []
				for item in tweet[index_t:index_t+ngram]:
					word = ''.join([x for x in item if ord(x) <128])		# Avoid problems with ascii values
					wordarray.append(word)
				wordstring = ' '.join(wordarray)
				if wordstring in tuple_array[index_n]:
					for i in range(index_t,index_t+ngram):
						listtweets[index_n][i] = True

		values = zip(*listtweets)

		new_tweet_array = []
		for index, word in enumerate(tweet):
			if ( any(values[index]) ):
				new_tweet_array.append(word)
		new_tweet = ' '.join(new_tweet_array)
		return new_tweet
			
	def splitbow(self, bow, ngramtypes_array):
		""" Split BOW in arrays of tuples with same length """
		splitted_bowkeys = []
		for ngram in ngramtypes_array:
			ngram_array = []
			for item in bow:
				if len(item) == ngram:
					ngram_array.append(item)
			test = helpers.unfold_tuples_strings(ngram_array)
			splitted_bowkeys.append(test)

		return splitted_bowkeys

	
	def start_naivebayes_classification(self, mode, ngram, minborder, maxborder, nr):
		""" Run Naive Bayes classifier """

		traintweets = []
		y_train = []
		testtweets = []
		y_test = []

		array = self.get_preprocessed_array(mode)


		# Initialize Bag of Words Object
		bowObject = BagOfWords(array, self.tweet_class, self.trainset)


		negbow, posbow = self.collect_bow(array, ngram, minborder, maxborder, nr/2)

		allwords = False
		if 'allwords' in mode:
			allwords = True
		self.train_tweetclasses, self.train_vectors = self.nb_create_traintestdata(array, ngram, posbow, negbow, self.trainset, mode, allwords=allwords)
		self.test_tweetclasses, self.test_vectors = self.nb_create_traintestdata(array, ngram, posbow, negbow, self.testset, mode, allwords=allwords)

		# Run Naive Bayes
		results = self.run_naivebayes(np.array(self.train_vectors), np.array(self.train_tweetclasses), np.array(self.test_vectors), np.array(self.test_tweetclasses),ngram, self.CROSS_VALIDATION)

		#results = self.run_naivebayes(np.array(train_tweets), np.array(train_classes), np.array(test_tweets), np.array(test_classes))

		return results


	def run_naivebayes(self, X_train, y_train, X_test, y_test,ngram, k):
		""" Fit Naive Bayes Classification on train set with cross validation. 
		Run Naive Bayes Classificaiton on test set. Return results
		"""

		# Transform train and test data
		count_vect = CountVectorizer(ngram_range=(ngram[0],ngram[len(ngram)-1]))

		X_train_counts = count_vect.fit_transform(X_train)
		tfidf_transformer = TfidfTransformer()
		X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

		X_new_counts = count_vect.transform(X_test)
		X_new_tfidf = tfidf_transformer.transform(X_new_counts)

		###print "** Fitting Naive Bayes classifier.."

		# Apply cross validation
		cv = cross_validation.KFold(X_train_tfidf.shape[0], n_folds=k, indices=True)

		cv_naivebayes = []
		for traincv, testcv in cv:
			clf_cv = MultinomialNB()
			clf_cv.fit(X_train_tfidf[traincv], y_train[traincv])
			y_test_cv, y_pred_cv = y_train[testcv], clf_cv.predict(X_train_tfidf[testcv])
			nb_tuple = (f1_score(y_test_cv, y_pred_cv), clf_cv)
			cv_naivebayes.append(nb_tuple)
		
		# Get best classifier
		(f1, best_clf) = max(cv_naivebayes,key=operator.itemgetter(0))
		

		###print "** Run Naive Bayes classifier.."
		y_true, y_pred = y_test, best_clf.predict(X_new_tfidf)

		tuples = precision_recall_fscore_support(y_true, y_pred)
		return (tuples, 'N.A.', 'N.A.')




	def compare_dummy_classification(self):
		""" Compares classifier to dummy classifiers. Return results"""
		X_train = self.train_vectors
		y_train = self.train_tweetclasses
		X_test = self.test_vectors
		y_test = self.test_tweetclasses

		dummy_results = []

		dummy = DummyClassifier(strategy='most_frequent',random_state=0)
		dummy.fit(X_train, y_train)
		y_true, y_preddum = y_test, dummy.predict(X_test)
		tuples = precision_recall_fscore_support(y_true, y_preddum)

		dummy1 = DummyClassifier(strategy='stratified',random_state=0)
		dummy1.fit(X_train, y_train)
		y_true, y_preddum1 = y_test, dummy1.predict(X_test)
		tuples1 = precision_recall_fscore_support(y_true, y_preddum1)

		dummy2 = DummyClassifier(strategy='uniform',random_state=0)
		dummy2.fit(X_train, y_train)
		y_true, y_preddum2 = y_test, dummy2.predict(X_test)
		tuples2 = precision_recall_fscore_support(y_true, y_preddum2)

		resulttuple = ('dummy freq', 'N.A.','N.A.', 'N.A.', 'N.A.', tuples)
		resulttuple1 = ('dummy strat', 'N.A.', 'N.A.', 'N.A.', 'N.A.', tuples1)
		resulttuple2 = ('dummy uni', 'N.A.', 'N.A.', 'N.A.', 'N.A.', tuples2)

		dummy_results.append(resulttuple)
		dummy_results.append(resulttuple1)
		dummy_results.append(resulttuple2)

		return dummy_results

	def svm_create_traintestdata(self, array, posbow, negbow, indexset,mode, **kwargs):
		""" creates dataset needed for training/testing of SVM"""
		vecformat_array = []
		vectest = []
		class_set = []
		tweet_class = []
		tweetarray = []
		for tweetindex in indexset:
			tweet = array[tweetindex]
			tweetarray.append(tweet)
			class_set.append(self.tweet_class[tweetindex])

		X_scaled = self.svm_create_vectorarray(tweetarray, posbow, negbow, self.scaler, mode)

		return (class_set, X_scaled)		

	def svm_create_vectorarray(self, array, posbow, negbow, scaler, mode):
		""" Create vector array for classification, scale appropriate"""
		vecformat_array = []
		for tweet in array:
			# Different vectormodes
			if ('pn-neutral' in mode):
				vec = self.tweet_to_vector_posnegneutral(tweet, posbow, negbow)
			if ('posneg' in mode):
				totalbow = dict(posbow.items() + negbow.items())
				vec =  self.tweet_to_vector(tweet, totalbow, True)
			if ('pos1' in mode):
				vec =  self.tweet_to_vector(tweet, posbow, True)
			if ('neg1' in mode):
				vec =  self.tweet_to_vector(tweet, negbow, True)
			if ('freq' in mode):
				totalbow = dict(posbow.items() + negbow.items())
				vec =  self.tweet_to_vector(tweet, totalbow, False)

			vecformat_array.append(vec)
		X = np.array(vecformat_array)

		# Scale train and test data with same scaler
		#scaler = kwargs.get('scaler', None)
		if self.scaler is None:
			scaler = preprocessing.StandardScaler().fit(X)
			self.scaler = scaler


		X_scaled = self.scaler.transform(X)  

		return X_scaled


	def collect_bow(self, array, ngram_types_array, posborder, negborder, nr):
		""" Collect Bag of words of trainset with specified array and ngrams
		Returns negative and positive bag of words
		"""
		bowObject = BagOfWords(array, self.tweet_class, self.trainset)
		negbow = {}
		posbow = {}

		# Create positive and negative bag of words
		for item in ngram_types_array:
			bowObject.create_corpus(item)
			posbow.update(bowObject.bow_partial(max_border=0+posborder, min_border=-1, nr=nr))
			negbow.update(bowObject.bow_partial(max_border=1, min_border=0+negborder, nr=nr))

		return (negbow, posbow)

	def get_preprocessed_array(self, arrayname):
		""" Get processed array according to name """
		mode = arrayname.split()

		if ( "stem" in mode[1]):
			return self.stemmed_tweets_array
		if ( "token" in mode[1]):
			return self.tokenized_tweets_array
		if ( "pos" in mode[1]): 
			return self.pos_tweets_array
		if ( "lemma" in mode[1]):
			return self.lemmatized_tweets_array
		else:
			return []

	def preprocess_tweets(self):
		""" Process tweets according to mode and set arrays """
		processObject = Preprocessing(self.mode, self.tweets)
		processObject.preprocess_tweets()
		if ( "stem" in self.mode):
			self.stemmed_tweets_array = processObject.stemmed_tweets_array
		if ( "token" in self.mode):
			self.tokenized_tweets_array = processObject.tokenized_tweets_array
		if ( "pos" in self.mode): 
			self.pos_tweets_array = processObject.pos_tweets_array
		if ( "lemma" in self.mode):
			self.lemmatized_tweets_array = processObject.lemmatized_tweets_array


	def create_sets(self):
		""" Create training/test/validation set via indices """
		debug = self.debug

		if (debug):
			try:
				print "reading from file"
				totallist = helpers.read_from_file(self.DEBUG_SETS)
				self.trainset = totallist[0]
				self.testset = totallist[1]
			except:
				print "! Error in reading from file debug.txt. Redo create_sets"
				debug = False
		if (not debug):
			for i in range(0, len(self.tweets)):
				# Test if random number is smaller than distribution for trainset
				r_nr = random.random()
				if (r_nr < self.distribution[0]):
					self.trainset.append(i)
				else:
					self.testset.append(i)
		if (self.dump):
			totallist = []
			totallist.append(self.trainset)
			totallist.append(self.testset)
			helpers.dump_to_file(self.DEBUG_SETS, totallist)			

	def print_sets(self):
		""" Print out training and test set"""
		print ">> Trainingset: (%d)" % len(self.trainset)
		print self.trainset
		print ">> Testset:  (%d)" % len(self.testset)
		print self.testset



	def count_classes(self):
		"""
		Counts and prings occurance of each class
		"""
		values = self.tweet_class.values()
		total = len(values)
	
		# Count occurances of classes
	 	activity_count = values.count(0)
		nonactivity_count = values.count(1)
		unknown_count = values.count(2)

		# Print
		print ">> Statistics:"
		print "Total number of tweets: %i" % total
		print "Total activity tweets: %i" % activity_count
		print "Total non-activity tweets: %i" % nonactivity_count
		print "Total unknown-activity tweets: %i" % unknown_count



	def tweet_to_vector(self, tweet_token, totalbow, boolean):
		""" Convert tweet to binary vector for occurances in BOW"""
		tweetstring = ' '.join(tweet_token)		# for comparison value tuples in string
		vec = []

		for index,x in enumerate(totalbow):
			if (' '.join(x) in tweetstring):
				if (boolean):
					vec.append(1.0)
				else:
					vec.append(totalbow[x])
			else:
				vec.append(0.0)

		return vec
			
				
	def tweet_to_vector_posnegneutral(self, tweet_token, posbow, negbow):
		""" Convert tweet to vector [posvalue, negvalue, neutralvalue]"""
		tweetstring = ' '.join(tweet_token)		# for comparison value tuples in string
		posvalue = 0
		negvalue = 0
		neutralvalue = 0

		# Get all single words from bag of words
		poswords = helpers.unfold_tuples(posbow)	
		negwords = helpers.unfold_tuples(negbow)		
		totalwords = list(set(poswords + negwords))

		for x in posbow:
			if (' '.join(x) in tweetstring):
				posvalue += posbow[x]
		for x in negbow:
			if (' '.join(x) in tweetstring):
				negvalue += negbow[x]
		for word in tweet_token:
			if ( word not in totalwords ):		# neutral words
				neutralvalue += 1

		vec  = [float(posvalue), float(negvalue), float(neutralvalue)]
		return vec


	def string_metrics(self, tuples):
		""" Create array of string values from values in tuples """
		(p,r,f, s) = tuples
		metriclist = [((f[0]+f[1])/2.0), f[0], f[1], ((p[0]+p[1])/2.0), p[0], p[1], ((r[0]+r[1])/2.0), r[0], r[1]]
		metrics_string_array = []
		for item in metriclist:
			metric = "%.4f" %item
			metrics_string_array.append(metric)

		return metrics_string_array

	def write_results_to_file(self, results):
		""" Write results to CSV file"""
		rows = []
		try:
			for item in results:
				mode, gamma, c, ngram, bow, tuples = item

				if isinstance(gamma, float):
					gamma = "%.4f" % gamma
				if isinstance(c, float):
					c = "%.0f" %c

				metriclist = self.string_metrics(tuples)
				row = [mode, gamma, c, ngram, bow]
				row += metriclist
				rows.append(row)
		except TypeError: 
			print "Error: Type of parameter result"

		helpers.write_to_csv(self.RESULTFILE, "a", rows)


	def write_begin(self):
		""" Write header for results to CSV file """
		rows = [["MODE","GAMMA", "C", "NGRAM", "BOW", "F1", "F1_0", "F1_1", "Precision", "P_0", "P_1", "Recall", "R_0", "R_1"]]
		helpers.write_to_csv(self.RESULTFILE, "wb", rows)


	def run_classification(self, modes, ngramarray, lenbows):
		""" Run classifications according to input and write results to file."""
		begin = time.time()

		# Run classifications according to parameters
		for mode in modes:
			print "-- RUN NEW MODE: %s.." % mode
			for ngram in ngramarray:
				print "-- RUN NEW NGRAM: %s.." % str(ngram)
				for lenbow in lenbows:
					resulttuple = None
					if 'svm' in mode:
							
						(result, gamma, c) = self.start_svm_classification(mode, ngram, 0,0, lenbow)
						resulttuple = [(mode, gamma, c, ngram, lenbow, result)]

					if 'nb' in mode:
						(result, gamma, c) = self.start_naivebayes_classification(mode, ngram, 0, 0, lenbow)
						resulttuple = [(mode, gamma, c, ngram, lenbow, result)]
					print "ONE"
					self.write_results_to_file(resulttuple)

		# Run dummy classification
		#dummy_result_array = self.compare_dummy_classification()
		#self.write_results_to_file(dummy_result_array)

		print "TIME TAKEN: %f seconds" % (time.time() - begin)


# call main with mode
m = Main(True, "frog lemma pos stem token --debug")
 
#classifiers = ['nb', 'svm']
#types_preprocess = ['token', 'stem', 'lemma', 'pos']

#['nb token posneg', 'nb token pos1', 'nb token neg1',
#DONE		'nb stem posneg', 'nb stem pos1', 'nb stem neg1',
#DONE		'nb lemma posneg', 'nb lemma pos1', 'nb lemma neg1',
#DONE		'nb pos posneg', 'nb pos pos1', 'nb pos neg1'
#DONE		'svm token pn-neutral' , 'svm token posneg', 'svm token pos1', 'svm token neg1','svm token freq'
#DONE		'svm stem pn-neutral' , 'svm stem posneg', 'svm stem pos1', 'svm stem neg1','svm stem freq',
#DONE		'svm lemma pn-neutral' , 'svm lemma posneg', 'svm lemma pos1', 'svm lemma neg1','svm lemma freq'
modes = ['svm lemma posneg' ]
#DONE 'svm pos posneg', 'svm pos pos1', 'svm pos neg1','svm pos freq']

#modes = ['nb token posneg']
ngramarray = [[1],[1,2], [1,2,3], [2,3]]
lenbows = [50, 74, 100, 124, 150, 174, 200]
#modes = ['svm token freq']
#ngramarray = [[1,2,3]]
#lenbows = [300, 500]
#m.write_begin()
m.run_classification(modes, ngramarray, lenbows)
