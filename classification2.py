import csv											# used for import csv tweets
import random										# used for train/test set creation
import os											# used for calling Frog in new terminal
import operator
import time											# used for timer
#from pynlpl.clients.frogclient import FrogClient	# used for Frog client
#import subprocess									# used for calling Frog in new terminal
#import signal										# used for calling Frog in new terminal
#import math	 
#import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_fscore_support

import helpers
from start_svm import Start_SVM
from start_nb import Start_NB
from preprocessing import Preprocessing
from bagofwords import BagOfWords


class Main(object):
	"""
	Class for blablabla
	"""
	# Input file reader
	DELIMITER = "\t"


	# Dictionary for class, classify unkown into non-activity
	class_dict = {"Y": 0, "N": 1, "U": 0}

	tweets = {}
	tweet_class = {}

	# Processed arrays
	stemmed_tweets_array = []
	tokenized_tweets_array = []
	pos_tweets_array = []
	lemmatized_tweets_array = []
	
	# File for classification results
	RESULTFILE = 'results_classification_26juni_rest.csv'

	TRAININGFILE = "2000test_annotated_v3.csv"

	# Arrays for classification
	train_tweetclasses = []
	train_vectors= []
	test_tweetclasses = []
	test_vectors = []

	# Cross validation folds
	CROSS_VALIDATION = 10

	def __init__(self, testingmode, mode):
		""" Initialize tweets, preprocess and create train/test sets"""
		self.mode = mode
  		#self.debug = "--debug" in mode
  		#self.dump = "--write" in mode
		self.debug = True
		self.dump = False
		self.testingmode = testingmode

		training_filename = self.TRAININGFILE.split('.')[0]
		self.initialize(self.TRAININGFILE)
		self.preprocess_tweets(self.mode,self.tweets, training_filename)
		if ( self.testingmode ):
			values = self.tweet_class.values()
			self.count_classes(values)

	def initialize(self, filename):
		""" Initializes tweet and class sets """
		print "** Initialize.."

		data = csv.reader(open(filename, 'rU'), delimiter=self.DELIMITER)
		for i, row in enumerate(data):
			if(i == 0):		# Ignore header
				pass
			else:
				# Get tweet and class of tweet 
				if (self.class_dict.get(row[5].upper()) is not None):
					self.tweets[i-1] = row[3]
					if ( self.testingmode ):
						self.tweet_class[i-1] = self.class_dict.get(row[5].upper())

		
	def get_activity_tweets(self, inputfile, mode, ngrambow, nr, loadclassifier):
		""" Extract new activity tweets from file"""
		# Create classifier on trainingdata of class
		(array, tuplebows) = self.setup_input_classification(mode, ngrambow, 0,0, nr)
		if 'svm' in mode:
			svmObject = Start_SVM(array, mode, self.tweet_class, True, tuplebows, self.CROSS_VALIDATION)
		if 'nb' in mode:
			nbObject = Start_NB(array, mode, self.tweet_class, True, tuplebows, ngrambow, self.CROSS_VALIDATION)
		# Get tweets of new data
		new_tweets = {}
		index = 0
		newdata = csv.reader(open(inputfile, 'rU'), delimiter=self.DELIMITER)
		for i, row in enumerate(newdata):
			if( row[5] == '' and row[0].isdigit()):
				new_tweets[index] = row[3]
				index += 1

		# Preprocess new data
		inputfile_filename = inputfile.split('.')[0]
		self.preprocess_tweets(mode,new_tweets, inputfile_filename)
		array = self.get_preprocessed_array(mode)

		# Classify newdata
		if 'svm' in mode:
			prediction = svmObject.start_classification(mode,array,  loadclassifier, 0.001, 10)
		if 'nb' in mode:
			prediction = nbObject.start_classification(mode,array,  False, loadclassifier)
	
		# Print to file
		self.count_classes(prediction.tolist())
		classification_filename = inputfile_filename + "_class.csv"
		helpers.write_classification_to_tweetfile(prediction,0, 5, inputfile, classification_filename)

	def analysis_classification(self, mode, ngrambow, nr, loadclassifier):
		""" Analyse classification of training & testdata"""
		DELIMITER = "\t"

		# Get tweets
		all_tweets = {}
		index = 0
		data = csv.reader(open(self.TRAININGFILE, 'rU'), delimiter=DELIMITER)
		for i, row in enumerate(data):
			if i == 0:
				pass
			else:
				all_tweets[index] = row[3]
				index += 1


		# Create classifier on trainingdata of class
		(array, tuplebows) = self.setup_input_classification(mode, ngrambow, 0,0, nr)
		if 'svm' in mode:
			svmObject = Start_SVM(array, mode, self.tweet_class, True, tuplebows, self.CROSS_VALIDATION)
		if 'nb' in mode:
			nbObject = Start_NB(array, mode, self.tweet_class, True, tuplebows, ngrambow, self.CROSS_VALIDATION)

		print "preprocess new data"
		# Preprocess new dataata
		training_filename = self.TRAININGFILE.split('.')[0]
		self.preprocess_tweets(mode,self.tweets, training_filename)
		array = self.get_preprocessed_array(mode)

		# Classify tweets
		if 'svm' in mode:
			prediction = svmObject.start_classification(mode,array, loadclassifier, 0.001, 10)
		if 'nb' in mode:
			prediction = nbObject.start_classification(mode,array,  False, loadclassifier)

		self.count_classes(prediction.tolist())
		classification_filename = training_filename + "_class.csv"
		helpers.write_classification_to_tweetfile(prediction,1, 7, self.TRAININGFILE, classification_filename)

	def start_svm_evaluation(self, array, mode, svmtype, ngrambow, minborder, maxborder, nr, tuplebows):
		""" Start SVM classification learning. Return results (resultscores_tuple, gamma1, c)"""
		svmObject = Start_SVM(array, mode, self.tweet_class, True, tuplebows, self.CROSS_VALIDATION)
		results = svmObject.start_svm_evaluation(mode, svmtype, minborder, maxborder, nr, tuplebows)
		return results

	def start_naivebayes_classification(self, array, mode, ngrambow, minborder, maxborder, nr, tuplebows):
		""" Start Naive Bayes classification learning. Return results (resultscores_tuple, N.A., N.A.)"""

		nbObject = Start_NB(array, mode, self.tweet_class, True, tuplebows, ngrambow, self.CROSS_VALIDATION)
		results = nbObject.start_naivebayes_evaluation(mode, minborder, maxborder, nr)

		return results


	def setup_input_classification(self, mode, ngrambow, minborder, maxborder, nr):
		""" Set up array and BOWs used for classification"""
		array = self.get_preprocessed_array(mode)
		tuplebows = self.collect_bow(array, ngrambow, minborder, maxborder, nr/2)
		return (array,tuplebows)


	def compare_dummy_classification(self):
		""" Compares classifier to dummy classifiers. Return results (resultscores_tuple, N.A., N.A.)"""
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
		""" Collect Bag of words of array with specified array and ngrams
		Returns negative and positive bag of words
		"""

		bowObject = BagOfWords(array, self.tweet_class)
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

		if ( "stem" in mode[2]):
			return self.stemmed_tweets_array
		if ( "token" in mode[2]):
			return self.tokenized_tweets_array
		if ( "pos" in mode[2]): 
			return self.pos_tweets_array
		if ( "lemma" in mode[2]):
			return self.lemmatized_tweets_array
		else:
			return []

	def preprocess_tweets(self, mode, tweets_dict, filename):
		""" Process tweets according to mode and set arrays """
		processObject = Preprocessing(mode, tweets_dict,filename)
		processObject.preprocess_tweets()
		if ( "stem" in mode):
			self.stemmed_tweets_array = processObject.stemmed_tweets_array
		if ( "token" in mode):
			self.tokenized_tweets_array = processObject.tokenized_tweets_array
		if ( "pos" in mode): 
			self.pos_tweets_array = processObject.pos_tweets_array
		if ( "lemma" in mode):
			self.lemmatized_tweets_array = processObject.lemmatized_tweets_array



	def count_classes(self, tweet_class_array):
		""" Counts occurance of each class
		"""
		total = len(tweet_class_array)
	
		# Count occurances of classes
	 	activity_count = tweet_class_array.count(0)
		nonactivity_count = tweet_class_array.count(1)

		# Print
		print ">> Statistics:"
		print "Total number of tweets: %i" % total
		print "Total activity tweets: %i" % activity_count
		print "Total non-activity tweets: %i" % nonactivity_count

	def string_metrics(self, f1_array):
		""" Create array of string values from values in tuples """
		metrics_string_array = []
		for item in f1_array:
			metric = "%.4f" %item
			metrics_string_array.append(metric)

		return metrics_string_array

	def write_results_to_file(self, results):
		""" Write results to CSV file"""
		rows = []
		try:
			for item in results:

				mode, gamma, c, ngram, bow, tuples = item
				f1_avg, f1_array = tuples

				if isinstance(gamma, float):
					gamma = "%.4f" % gamma
				if isinstance(c, float):
					c = "%.0f" %c

				f1_avg_4f = "%.4f" % f1_avg

				metriclist = self.string_metrics(f1_array)
				row = [mode, gamma, c, ngram, bow, f1_avg_4f]
				row += metriclist
				rows.append(row)
		except TypeError: 
			print "Error: Type of parameter result"
			print results

		helpers.write_to_csv(self.RESULTFILE, "a", rows)


	def write_begin(self):
		""" Write header for results to CSV file """
		# Create headers for rounds
		list_roundnr = []
		for i in range(1, self.CROSS_VALIDATION+1):
			roundnr_string = "Round %i" %i
			list_roundnr.append(roundnr_string)
			
		headers = [["MODE","GAMMA", "C", "NGRAM", "BOW", "F1 AVG"]]
		rows = [ headers[0]+list_roundnr ]
		helpers.write_to_csv(self.RESULTFILE, "wb", rows)


	def run_classification_evaluation(self, modes, ngramarray, lenbows):
		""" Run classifications according to input and write results to file."""
		begin = time.time()


		# Run classifications according to parameters
		for mode in modes:
			print "-- RUN NEW MODE: %s.." % mode
			for ngram in ngramarray:
				print "-- RUN NEW NGRAM: %s.." % str(ngram)
				for lenbow in lenbows:
					#try:
					resulttuple = None
					array, tuplebow = self.setup_input_classification(mode, ngram, 0, 0, lenbow)
					if 'svm' in mode:
						svmtype = mode.split()[1]
						(result, gamma, c) = self.start_svm_evaluation(array, mode, svmtype, ngram, 0,0, lenbow,tuplebow)
						resulttuple = [(mode, gamma, c, ngram, lenbow, result)]
					if 'nb' in mode:
						(result, gamma, c) = self.start_naivebayes_classification(array,mode, ngram, 0, 0, lenbow, tuplebow)
						resulttuple = [(mode, gamma, c, ngram, lenbow, result)]

					#except Exception:
					#	print "PROBLEM OCCURED in mode: %s, ngram: %s, lenbow: %i" %(mode,str(ngram), lenbow)
					#	print "A"
					self.write_results_to_file(resulttuple)

		# Run dummy classification
		#dummy_result_array = self.compare_dummy_classification()
		#self.write_results_to_file(dummy_result_array)

		print "TIME TAKEN: %f seconds" % (time.time() - begin)




# call main with mode
m = Main(True, "frog lemma pos stem token --debug")
#m.get_activity_tweets('day_saturday.csv','svm lemma posneg --debug --write', [1,2], 100, True)
#m.write_begin()
#m.analysis_classification('nb lemma posneg --debug', [1,2], 100, False)

#Dmodes = ['nb token posneg', 'nb token pos1', 'nb token neg1',
#D		'nb stem posneg', 'nb stem pos1', 'nb stem neg1',
#D		'nb lemma posneg', 'nb lemma pos1', 'nb lemma neg1',
#D		'nb pos posneg', 'nb pos pos1', 'nb pos neg1']
#modes = ['svm token posneg', 'svm token pos1', 'svm token neg1','svm token freq']
#DONE		'svm token pn-neutral' , 'svm token posneg', 'svm token pos1', 'svm token neg1','svm token freq'
#DONE		'svm stem pn-neutral' , 'svm stem posneg', 'svm stem pos1', 'svm stem neg1','svm stem freq',
#modes= ['svm lemma pn-neutral' , 'svm lemma posneg', 'svm lemma pos1', 'svm lemma neg1','svm lemma freq',
#		'svm pos posneg', 'svm pos pos1', 'svm pos neg1','svm pos freq']
#DONE ['svm token freq', 'svm stem freq', 'svm lemma freq','svm pos freq']
#TODO: modes = ['svm lemma pn-neutral', svm pos pn-neutral

"""
ALL POSSIBLE MODES:
modes = {svm ln, svm rb} x {token, stem, lemma, pos} x{posneg, pos1, neg1, freq} x {[1], [1,2], [1,2,3], [2,3]} x {50, 74, 100, 124, 150, 174, 200}
	+ {nb} x {token, stem, lemma, pos} x {posneg, pos1, neg1}  x {[1], [1,2], [1,2,3], [2,3]} x {50, 74, 100, 124, 150, 174, 200}
	= 896+336

['svm token posneg', 'svm token pos1', 'svm token neg1','svm token freq',
		 'svm stem posneg', 'svm stem pos1', 'svm stem neg1','svm stem freq',
		'svm lemma posneg', 'svm lemma pos1', 'svm lemma neg1','svm lemma freq',
		'svm pos posneg', 'svm pos pos1', 'svm pos neg1','svm pos freq']
"""
modes= ['svm rbf stem posneg', 'svm rbf stem pos1', 'svm rbf stem neg1']
ngramarray = [[1],[1,2], [1,2,3], [2,3]]
lenbows = [50, 74, 100, 124, 150, 174, 200]
m.run_classification_evaluation(modes, ngramarray, lenbows)
#

modes = [ 'svm rbf token posneg']
ngramarray = [[1,2,3]]
lenbow = [150,174,200]
m.run_classification_evaluation(modes, ngramarray, lenbows)
#
ngramarray = [[2,3]]
lenbows = [50, 74, 100, 124, 150, 174, 200]
m.run_classification_evaluation(modes, ngramarray, lenbows)
#m.write_begin()
modes = ['svm ln pos posneg', 'svm ln pos pos1', 'svm ln pos neg1','svm lnpos freq']
ngramarray = [[1],[1,2], [1,2,3], [2,3]]
m.run_classification_evaluation(modes, ngramarray, lenbows)

