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
from start_nb import Start_NB


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
		tuplebows = self.collect_bow(array, ngrambow, minborder, maxborder, nr/2)

		svmObject = Start_SVM(array, self.tweet_class, self.trainset, self.testset, True, tuplebows, self.CROSS_VALIDATION)
		results = svmObject.start_svm_testing(mode, minborder, maxborder, nr)

		return results


	
	def start_naivebayes_classification(self, mode, ngrambow, minborder, maxborder, nr):
		""" Run Naive Bayes classifier """

		array = self.get_preprocessed_array(mode)
		tuplebows = self.collect_bow(array, ngrambow, minborder, maxborder, nr/2)

		nbObject = Start_NB(array, self.tweet_class, self.trainset, self.testset, True, tuplebows, ngrambow, self.CROSS_VALIDATION)
		results = nbObject.start_naivebayes_testing(mode, minborder, maxborder, nr)
		return results


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
modes = ['nb lemma posneg' ]
#DONE 'svm pos posneg', 'svm pos pos1', 'svm pos neg1','svm pos freq']

#modes = ['nb token posneg']
ngramarray = [[1],[1,2], [1,2,3], [2,3]]
lenbows = [50, 74, 100, 124, 150, 174, 200]
#modes = ['svm token freq']
#ngramarray = [[1,2,3]]
#lenbows = [300, 500]
#m.write_begin()
m.run_classification(modes, ngramarray, lenbows)
