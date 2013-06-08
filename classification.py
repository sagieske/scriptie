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



import helpers



class Main(object):
	"""
	Class for blablabla
	"""
	# Read file
	DELIMITER = "\t"
	data = csv.reader(open("2000test_annotated.csv", 'rU'), delimiter=DELIMITER)

	# Dictionary for class, classify unkown into non-activity
	class_dict = {"Y": 0, "N": 1, "U": 1}

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

	train_tweetclasses = []
	train_vectors= []
	test_tweetclasses = []
	test_vectors = []

	def __init__(self, mode):
		""" Set up for training"""
		self.mode = mode
  		self.debug = "--debug" in mode
  		self.dump = "--write" in mode

		self.initialize()
		self.preprocess_tweets()
		self.create_sets()
		self.count_classes()

	def initialize(self):
		""" Initializes tweet and class sets """
		print "** Initialize.."
		for i, row in enumerate(self.data):
			# Ignores header
			if(i == 0):
				pass
			else:
			# TEMP, for testing only
				begin = 1
				end = 2000
				if (i >= begin and i <= end):
					# TEMP, Only add if class is known! for testing only
					if (self.class_dict.get(row[5].upper()) is not None):
						# Get tweet and class 
						self.tweets[i-begin] = row[3]
						self.tweet_class[i-begin] = self.class_dict.get(row[5].upper())
					else:
						# Get tweet and class 
						self.tweets[i-begin] = row[3]
						self.tweet_class[i-begin] = 1

	def run(self, mode, in_ngrambow, in_minborder, in_maxborder, nr):
		""" Runs classification learning"""
		# Create BOW
		array = self.get_preprocessed_array(mode)
		ngram_forbow = in_ngrambow
		minborder = in_minborder
		maxborder = in_maxborder
		negbow, posbow = self.collect_bow(array, ngram_forbow, minborder, maxborder, nr)
		
		# Create train & test data (classes, vectors)
		self.train_tweetclasses, self.train_vectors, trainscaler = self.create_traintestdata(array, posbow, negbow, self.trainset, mode)
		self.test_tweetclasses, self.test_vectors, trainscaler = self.create_traintestdata(array, posbow, negbow, self.testset, mode, scaler=trainscaler)

		# Run SVM
		results = self.configure_svm2(np.array(self.train_vectors), np.array(self.train_tweetclasses), np.array(self.test_vectors), np.array(self.test_tweetclasses), 1, mode)
		return results
		#print "TEST"
		#print teststring

	def configure_svm2(self, X_train, y_train, X_test, y_test, k,mode):
		""" Configures parameters for svm """
		clf = svm.SVC(kernel='linear', C=1)
		# Grid for testing
		param_grid = [
		 {'C': np.logspace(1,5,5), 'gamma': np.logspace(-3,0,5), 'kernel': ['rbf']}
		# {'C': np.logspace(1,5,10), 'gamma': np.logspace(-3,0,10), 'kernel': ['linear']}
		]


		print "# Tuning hyper-parameters F1\n"
		score_func = metrics.f1_score

		clf = GridSearchCV(SVC(), param_grid, score_func=score_func,  n_jobs=-1 )
		clf.fit(X_train, y_train, cv=5)
		#print "Best parameters set found on development set:"
		#print clf.best_params_
		dict_param = clf.best_params_
		gamma1 = dict_param['gamma']
		c = dict_param['C']

		y_true, y_pred = y_test, clf.predict(X_test)
		
		tuples = precision_recall_fscore_support(y_true, y_pred)
		#self.string_metrics(tuples)
		#results = [('stem special', [1,2], 10, tuples), ('lemma special', [1,2,3], 10, tuples), ('lemma pos1', [1], 10, tuples)]
		#self.write_results_to_file(results)
		return (tuples, gamma1, c)

		#self.compare_dummy(clf, X_train, y_train, X_test, y_test)


	def compare_dummy(self):
		""" Compares classifier to dummy classifiers"""
		#print "\nDetailed classification report:\n"
		#print "The model is trained on the full development set.\n"
		#print "The scores are computed on the full evaluation set.\n"

		X_train = self.train_vectors
		y_train = self.train_tweetclasses
		X_test = self.test_vectors
		y_test = self.test_tweetclasses

		dummy = DummyClassifier(strategy='most_frequent',random_state=0)
		dummy.fit(X_train, y_train)
		y_true, y_preddum = y_test, dummy.predict(X_test)
		print "Dummy score most frequent \n"
		print classification_report(y_true, y_preddum)

		dummy1 = DummyClassifier(strategy='stratified',random_state=0)
		dummy1.fit(X_train, y_train)
		y_true, y_preddum1 = y_test, dummy1.predict(X_test)
		print "Dummy score stratified \n"
		print classification_report(y_true, y_preddum1)

		dummy2 = DummyClassifier(strategy='uniform',random_state=0)
		dummy2.fit(X_train, y_train)
		y_true, y_preddum2 = y_test, dummy2.predict(X_test)
		print "Dummy score unified\n"
		print classification_report(y_true, y_preddum2)
		

	def create_traintestdata(self, array, posbow, negbow, indexset,mode, **kwargs):
		""" creates dataset needed for training/testing"""
		vecformat_array = []
		vectest = []
		class_set = []
		tweet_class = []
		for tweetindex in indexset:
			tweet = array[tweetindex]
			class_set.append(self.tweet_class[tweetindex])

			# Different vectormodes
			if ('special' in mode):
				vec = self.tweet_to_specialvector(tweet, posbow, negbow)
				#vecformat = self.specialvector_to_format(vec, tweet_class)	
			if ('posneg' in mode):
				totalbow = dict(posbow.items() + negbow.items())
				vec =  self.tweet_to_vector(tweet, totalbow)
				#vecformat = self.specialvector_to_format(vec)
			if ('pos1' in mode):
				vec =  self.tweet_to_vector(tweet, posbow)
				#vecformat = self.specialvector_to_format(vec)
			if ('neg1' in mode):
				vec =  self.tweet_to_vector(tweet, negbow)

			vecformat = self.specialvector_to_format(vec, tweet_class)	
			#class_set.append(tweet_class)
			vecformat_array.append(vecformat)

		X = np.array(vecformat_array)

		scaler = kwargs.get('scaler', None)
		if scaler is None:
			scaler = preprocessing.StandardScaler().fit(X)


		X_scaled = scaler.transform(X)  
		#scaler = None
		return (class_set, X_scaled, scaler)


	def collect_bow(self, array, ngram_types_array, posborder, negborder, nr):
		""" Collect Bag of words of trainset with specified array and ngrams
		Returns negative and positive bag of words
		"""
		bowObject = BagOfWords(array, self.tweet_class, self.trainset)
		totalbow = {}
		negbow = {}
		posbow = {}

		# Create positive and negative bag of words
		for item in ngram_types_array:
			bowObject.create_corpus(item)
			#bowObject.print_topcorpus(bowObject.bow_partial(max_border=0+posborder, min_border=-1, nr=20),20)
			posbow.update(bowObject.bow_partial(max_border=0+posborder, min_border=-1, nr=nr))
			negbow.update(bowObject.bow_partial(max_border=1, min_border=0+negborder, nr=nr))

		return (negbow, posbow)

	def get_preprocessed_array(self, arrayname):
		""" Get processed array according to name """
		if ( "stem" in arrayname):
			return self.stemmed_tweets_array
		if ( "token" in arrayname):
			return self.tokenized_tweets_array
		if ( "pos" in arrayname): 
			return self.pos_tweets_array
		if ( "lemma" in arrayname):
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
				totallist = self.read_from_file(self.DEBUG_SETS)
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

	def read_from_file(self, filename):
		"""	Load array from file """
		f = file(filename, "r")
		array = pickle.load(f)
		return array

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



	def specialvector_to_format(self, vec, tweet_class):
		""" Convert vector to libsvm format """
		vecformat = []
		#vecformat.append(tweet_class)
		#valuelist = bow.values()
		# libsvm only needs features with non-zero value
		for index in range(0,len(vec)):
			vecformat.append(vec[index])
		return vecformat

	def tweet_to_vector(self, tweet_token, totalbow):
		tweetstring = ' '.join(tweet_token)		# for comparison value tuples in string
		vec = []
		for index,x in enumerate(totalbow):
			if (' '.join(x) in tweetstring):
				vec.append(1.0)
			else:
				vec.append(0.0)
		return vec
			
				
	def tweet_to_specialvector(self, tweet_token, posbow, negbow):
		""" Convert tweet to posnegneutral vector"""
		tweetstring = ' '.join(tweet_token)		# for comparison value tuples in string
		posvalue = 0
		negvalue = 0
		neutralvalue = 0

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
			if ( word not in totalwords ):
				neutralvalue += 1
		# TODO: differences??
		#vec  = [float(posvalue/len(totalwords)), float(negvalue/len(totalwords)), float(neutralvalue/float(len(totalwords)))]
		vec  = [float(posvalue), float(negvalue), float(neutralvalue)]
		return vec

	def print_information(self, confusionmatrix,tuples):
		tp = confusionmatrix[0][0]
		fp = confusionmatrix[0][1]
		fn = confusionmatrix[1][0]
		tn = confusionmatrix[1][1]
		(p,r,f, s) = tuples
		precision = tp/float(tp + fp)
		recall = tp/float(tp+fn)
		accuracy = (tp +tn) / float(tp +tn+fp+fn)
		f1 = 2 * (precision * recall) /float(precision + recall)
		print "Confusion matrix:"
		print confusionmatrix
		print "Precision: %f\n Recall: %f\n Accuracy: %f \nF-1: %f\n" %(precision, recall, accuracy, f1)
		print "-- Precision: %f\n Recall: %f\nF-1: %f\n" %(p[0], r[0], f[0])
		print "-- Precision: %f\n Recall: %f\nF-1: %f\n" %(p[1], r[1], f[1])
		print "-- Precision: %f\n Recall: %f\nF-1: %f\n" %(((p[0]+p[1])/2.0), ((r[0]+r[1])/2.0), ((f[0]+f[1])/2.0))
		self.string_metrics(tuples)
		self.write_results_to_file(results)

	def string_metrics(self, tuples):
		""" Creates string from values in tuple"""
		(p,r,f, s) = tuples
		metriclist = [((f[0]+f[1])/2.0), f[0], f[1], ((p[0]+p[1])/2.0), p[0], p[1], ((r[0]+r[1])/2.0), r[0], r[1]]
		string = ''
		for item in metriclist:
			string+= "%.4f".ljust(8) %item

		return string

	def write_results_to_file(self, results):
		#string = 'MODE'.ljust(15) + 'GAMMA'.ljust(9) + 'C'.ljust(5) + 'NGRAM'.ljust(10) + 'BOW'.ljust(10)+ 'F1'.ljust(10) +'F1_0'.ljust(10)+'F1_1'.ljust(10) + 'Precision'.ljust(10)+ 'P_0'.ljust(10) + 'P_1'.ljust(10) + 'Recall'.ljust(10) + 'R_0'.ljust(10) + 'R_1'.ljust(10) + '\n'
	 	string = ''
		for item in results:
			mode, gamma1, c1, ngram, bow, tuples = item
			gamma = "%.4f".ljust(7) % gamma1
			c = "%.0f".ljust(7) %c1
			resultstring = self.string_metrics(tuples)
			string += mode.ljust(15) + gamma + c + str(ngram).ljust(10) + str(bow).ljust(10) +  resultstring + "\n"

		f = open('results7juni.txt', 'a')
		f.write(string)
		f.close()

	def write_begin(self):
		string = 'MODE'.ljust(15) + 'GAMMA'.ljust(9) + 'C'.ljust(5) + 'NGRAM'.ljust(10) + 'BOW'.ljust(10)+ 'F1'.ljust(10) +'F1_0'.ljust(10)+'F1_1'.ljust(10) + 'Precision'.ljust(10)+ 'P_0'.ljust(10) + 'P_1'.ljust(10) + 'Recall'.ljust(10) + 'R_0'.ljust(10) + 'R_1'.ljust(10) + '\n'
		f = open('results7juni.txt', 'w+')
		f.write(string)
		f.close()


# call main with mode
m = Main("frog lemma pos stem token --debug")
"""
modes = ['token special', 'token posneg', 'token pos1', 'token neg1', 'stem special', 'stem posneg', 'stem pos1', 'stem neg1', 'lemma special', 'lemma posneg', 'lemma pos1', 'lemma neg1', 'pos special', 'pos posneg', 'pos pos1', 'pos neg1',]
ngramarray = [[1], [2], [1,2], [1,2,3], [2,3]]
lenbows = [10,20,25,30,40,50]
resultarray = []
for mode in modes:
	for ngram in ngramarray:
		for lenbow in lenbows:
			results = m.run(mode, ngram, 0,0, lenbow)
			(result, gamma, C) = m.run(mode, ngram, 0,0, lenbow)
			resulttuple = (mode, gamma, C, ngram, lenbow, results)
			resultarray.append(resulttuple)

m.write_results_to_file(resultarray)
"""
begin = time.time()

modes = ['token special', 'token posneg', 'token pos1', 'token neg1', 'stem special', 'stem posneg', 'stem pos1', 'stem neg1', 'lemma special', 'lemma posneg', 'lemma pos1', 'lemma neg1', 'pos special', 'pos posneg', 'pos pos1', 'pos neg1']
ngramarray = [[1], [2], [1,2], [1,2,3], [2,3]]
lenbows = [10,20,25,30,40,50]
m.write_begin()

for mode in modes:
	for ngram in ngramarray:
		for lenbow in lenbows:
			(result, gamma, c) = m.run(mode, ngram, 0,0, lenbow)
			print "RESULTS"
			resulttuple = [(mode, gamma, c, ngram, lenbow, result)]
			m.write_results_to_file(resulttuple)

"""
ngramarray = [[1]]
lenbows = [10]
resultarray = []
for mode in modes:
	for ngram in ngramarray:
		for lenbow in lenbows:
			(result, gamma, C) = m.run(mode, ngram, 0,0, lenbow)
			print "RESULTS"
			resulttuple = (mode, gamma, C, ngram, lenbow, results)
			resultarray.append(resulttuple)
"""
"""
(tuples, gamma, c) = m.run("token posneg", [1], 0,0, 10)
resultarray = [('token special', gamma, c, [[1]], 10, tuples)]
m.write_begin()

m.write_results_to_file(resultarray)
m.write_results_to_file(resultarray)
"""
print "TIME TAKEN: %f seconds" % (time.time() - begin)
#m.compare_dummy()
