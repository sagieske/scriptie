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

	def __init__(self, mode):
		""" Set up for training"""
		self.mode = mode
  		self.debug = "--debug" in mode
  		self.dump = "--write" in mode

		self.initialize()
		self.preprocess_tweets()
		self.create_sets()
		#self.print_sets()
		self.count_classes()

		# Dump sets
		"""
		if (self.dump):
			totallist = []
			totallist.append(self.trainset)
			totallist.append(self.testset)
			self.write_to_file(self.DEBUG_SETS, totallist)
		"""	

		#def run(self, svm_mode):

		#b.create_corpus(2)
		#bow3 = b.bow_partial(max_border=0.3, min_border=-0.3, nr=10)

		#totalbow.update(bow3)

		# TRAIN SVM
		#self.create_traindata(self.lemmatized_tweets_array, totalbow, posbow, negbow)

	def run(self, mode):
		""" Runs classification learning"""
		# Create BOW
		array = self.get_preprocessed_array(mode)
		negbow, posbow = self.collect_bow(array, [1,2], 0,0)
		
		# Create train & test data (classes, vectors)
		train_tweetclasses, train_vectors = self.create_traintestdata(array, posbow, negbow, self.trainset, mode)
		test_tweetclasses, test_vectors = self.create_traintestdata(array, posbow, negbow, self.testset, mode)
	
		print "SVM"
		self.configure_svm(train_vectors, train_tweetclasses, test_vectors, test_tweetclasses, 1)

		# Train SVM
		#self.try_svm2(np.array(train_vectors), np.array(train_tweetclasses), np.array(test_vectors), np.array(test_tweetclasses))
		#self.try_svm_cross(np.array(train_vectors), np.array(train_tweetclasses), np.array(test_vectors), np.array(test_tweetclasses))

	def configure_svm(self, X_train, y_train, X_test, y_test, k):
		""" Configures parameters for svm """
		clf = svm.SVC(kernel='linear', C=1)
		param_grid = [
		 # {'C': np.logspace(1,5,10), 'gamma': np.logspace(-10, 2, 14),'kernel': ['linear']},
		  #{'C': np.logspace(1,5,5), 'gamma': np.logspace(-5, 2, 5), 'kernel': ['rbf']},
			{'C': np.array([10,100,1000]), 'gamma': np.array([0.0001, 0.001, 0.005])},
		 ]
		scores = [
			('precision', precision_score),
			('recall', recall_score),
			('accuracy', accuracy_score),
			('f1', f1_score)
		]


		print "TEST RUN"
		for score_name, score_func in scores:
			print "# Tuning hyper-parameters for %s\n" % score_name
			clf = GridSearchCV(SVC(C=1), param_grid, score_func=score_func)
			print "Fitting..."
			clf.fit(X_train, y_train, cv=k)
			print "Best parameters set found on development set:\n"
			print clf.best_estimator_
			print "\nGrid scores on development set:\n"
			for params, mean_score, scores in clf.grid_scores_:
				print "%0.3f (+/-%0.03f) for %r" % (
				    mean_score, scores.std() / 2, params)

			print "\nDetailed classification report:\n"
			print "The model is trained on the full development set.\n"
			print "The scores are computed on the full evaluation set.\n"
			y_true, y_pred = y_test, clf.predict(X_test)
			print self.classification_report(y_true, y_pred)

	def classification_report(y_true, y_pred):
		counter_true = 0
		counter_false = 1
		for i in range(0,len(y_true)):
			if y_true == y_pred:
				count_true += 1
			else:
				count_false = 1
		 
		accuracy = accuracy_score(y_true, y_pred)
		return "FINAL TEST: correct: %f, incorrect: %f, Accuracy: %f\n" %(count_true, count_false, accuracy)


	def create_traintestdata(self, array, posbow, negbow, indexset,mode):
		""" creates dataset needed for training/testing"""
		vecformat_array = []
		vectest = []
		class_set = []
		tweet_class = []
		for tweetindex in indexset:
			tweet = array[tweetindex]
			vectest.append(self.tweet_class[tweetindex])
			tweet_class.append(vectest)
			#tweet_class.append(self.tweet_class[tweetindex])

			# Different vectormodes
			if ('special' in mode):
				vec = self.tweet_to_specialvector(tweet, posbow, negbow)
				vecformat = self.specialvector_to_format(vec, tweet_class)	
			else:
				totalbow = dict(posbow.items() + negbow.items())
				vec =  self.tweet_to_vector(tweet, totalbow)
				vecformat = self.vector_to_format2(vec)

			class_set.append(tweet_class)
			vecformat_array.append(vecformat)

		return (class_set, vecformat_array)

		"""
		def create_traindata(self, array,bow, posbow, negbow):
			# TODO: ARRAY DEPENDENT ON PREPROCESSING?!
		""" #Writes training data to file with libsvm format: <label> <feature-id>:<feature-value> <feature-id>:<feature-value>
		"""
			vecformat_array = []
			vecformattest_array = []
			tweettrain = []
			tweettest = []
			for tweetindex in self.trainset:
				# get tweet
				tweet = array[tweetindex]
				tweet_class = self.tweet_class[tweetindex]
				vec =  self.tweet_to_vector(tweet, bow)
				vecformat = self.vector_to_format(vec, bow, tweet_class)
				vecformat2 = self.vector_to_format2(vec)
				specialvec = self.tweet_to_specialvector(tweet, posbow, negbow)
				specialvecformat = self.specialvector_to_format(specialvec, tweet_class)

				tweettrain.append(tweet_class)
				vecformat_array.append(vecformat2)
				#vecformat_array.append(specialvecformat)


			for tweetindex in self.testset:
				# get tweet
				tweet = array[tweetindex]
				tweet_class = self.tweet_class[tweetindex]
				vec =  self.tweet_to_vector(tweet, bow)
				vecformat2 = self.vector_to_format2(vec)
				specialvec = self.tweet_to_specialvector(tweet, posbow, negbow)
				specialvecformat = self.specialvector_to_format(specialvec, tweet_class)
				tweettest.append(tweet_class)
				vecformattest_array.append(vecformat2)
				#vecformattest_array.append(specialvecformat)

			print len(tweettrain)
			print len(tweettest)
			self.try_svm2(np.array(vecformat_array), np.array(tweettrain), np.array(vecformattest_array), np.array(tweettest))
			self.try_svm_cross(np.array(vecformat_array), np.array(tweettrain), np.array(vecformattest_array), np.array(tweettest))
		"""

	def collect_bow(self, array, ngram_types_array, posborder, negborder):
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
			posbow.update(bowObject.bow_partial(max_border=0+posborder, min_border=-1, nr=20))
			negbow.update(bowObject.bow_partial(max_border=1, min_border=0+negborder, nr=20))

		return (negbow, posbow)

		"""
		test_bow = BagOfWords(self.lemmatized_tweets_array, self.tweet_class, self.trainset)
		test_bow.create_corpus(1)
		posbow= test_bow.bow_partial(max_border=0, min_border=-1, nr=20)
		negbow= test_bow.bow_partial(max_border=1, min_border=0, nr=20)
		test_bow2 = BagOfWords(self.lemmatized_tweets_array, self.tweet_class, self.trainset)
		test_bow2.create_corpus(2)
		posbow.update(test_bow2.bow_partial(max_border=0, min_border=-1, nr=20))
		negbow.update(test_bow2.bow_partial(max_border=1, min_border=0, nr=20))
		#totalbow= b.bow_partial(max_border=0.1, min_border=-0.1, nr=20)
		"""

	def initialize(self):
		""" Initializes tweet and class sets """
		print "** Initialize.."
		for i, row in enumerate(self.data):
			# Ignores header
			if(i == 0):
				pass
			else:
			# TEMP, for testing only
				begin = 1000
				end = 1999
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


	def try_svm2(self, X_train, y_train, X_test, y_test):
		#X_train, y_train = load_svmlight_file("train_dataset.txt")
		clf=  svm.SVC(C=10000, gamma= 0.0006)
		clf.fit(X_train, y_train)  
		print clf

		counter_good = 0
		counter_bad = 0
		#prediction = clf.predict(X_train)
		y_pred = []

		for index in range(0,len(X_test)):

			prediction = clf.predict(X_test[index])	

			y_pred.append(prediction[0])
			if (prediction[0] == y_test[index]):
				counter_good += 1
			else:
				counter_bad +=1

		print "correct: %d" %counter_good
		print "NOT correct: %d" %counter_bad

		#y_pred =np.array(y_pred)
		accuracy = accuracy_score(y_test, y_pred)
		print "Accuracy: %f" %accuracy

	def try_svm_cross(self, X_train, y_train, X_test, y_test):
		k_fold = cross_validation.KFold(len(X_train), n_folds=3, indices=True)
		#for train_indices, test_indices in k_fold:
		#	print 'Train: %s | test: %s' % (train_indices, test_indices)
		svc = svm.SVC()
		#[svc.fit(X_train[train], y_train[train]).score(X_train[test], y_train[test]) for train, test in k_fold]
		#print cross_validation.cross_val_score(svc, X_train, y_train, cv=k_fold, n_jobs=-1)

		#clf = GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas), n_jobs=-1)
 		#clf.fit(X_train, y_train)
		#print clf.best_score_


		print "Test cparam"
		cparam = np.logspace(1, 10, 10)
		gammas = np.logspace(-10, 3, 10)
		clf = GridSearchCV(estimator=svc, param_grid=dict(C=cparam, gamma=gammas), n_jobs=-1)
 		clf.fit(X_train, y_train)
		print clf.best_score_
		print "gamma: %f" % clf.best_estimator_.gamma
		print "C: %f" %  clf.best_estimator_.C


	def get_preprocessed_array(self, arrayname):
		""" Get processed array according to name """
		if ( "stem" in arrayname):
			return self.stemmed_tweets_array
		if ( "token" in arrayname):
			return self.tokenized_tweets_array
		if ( "pos" in arrayname): 
			return self.pos_tweets_array
		if ( "lemma" in sarrayname):
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

	def print_sets(self):
		print ">> Trainingset: (%d)" % len(self.trainset)
		print self.trainset
		print ">> Testset:  (%d)" % len(self.testset)
		print self.testset


	def write_to_file(self, filename, array):
		"""	Dump array to file """
		f = file(filename, "w")
		pickle.dump(array, f)

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


		


	def write_traindata(self, vecformatdata):
		""" Write traindata to file """
		f = open('train_dataset.txt','w')

		for vector in vecformatdata:
			for key,value in sorted(vector.items()):
				if (key == 0):
					string = "%d" % (value)
				else:
					substring = "%d:%1.10f" % (key,value)
					string += " " + substring
			f.write(string+'\n')

		f.close()

	def tweet_to_vector(self, tweet_token, bow):
		""" Convert tweet to boolean vector"""
		tweetstring = ' '.join(tweet_token)		# for comparison value tuples in string
		vec = [ (' '.join(x) in tweetstring) for x in bow]
		return vec

	def vector_to_format(self, vec, bow, tweet_class):
		""" Convert vector to libsvm format """
		vecformat = {}
		vecformat[0] = tweet_class
		valuelist = bow.values()
		# libsvm only needs features with non-zero value
		for index in range(0,len(vec)):
			if vec[index] == True:
				vecformat[index+1] = valuelist[index]
		return vecformat

	def vector_to_format2(self, vec):
		""" Convert vector to libsvm format """
		vecformat = []
		# libsvm only needs features with non-zero value
		for index in range(0,len(vec)):
			if vec[index] == True:
				vecformat.append(1)
			else:
				vecformat.append(0)
		return vecformat

	def specialvector_to_format(self, vec, tweet_class):
		""" Convert vector to libsvm format """
		vecformat = []
		#vecformat.append(tweet_class)
		#valuelist = bow.values()
		# libsvm only needs features with non-zero value
		for index in range(0,len(vec)):
			vecformat.append(vec[index])
		return vecformat

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
		for index in totalwords:
			if ( index not in tweetstring ):
				neutralvalue += 1
		vec  = [float(posvalue/len(totalwords)), float(negvalue/len(totalwords)), float(neutralvalue/float(len(totalwords)))]
		return vec


# call main with mode
m = Main("frog lemma pos stem token --debug")
m.run("stem special")




##### TESTING
#specialvec = m.tweet_to_specialvector(['ik', 'ga', 'vanavond', 'dit'], {('ik',):0.4, ('vanavond', 'dit'):0.2}, {('dit',):-0.2})
#m.specialvector_to_format(specialvec, 0)
#m.create_traindata([['ik','ga','vanavond','naar', 'de', 'stad'], ['kijk', 'vanavond', 'om', 'tv']], {('kijk',):-1, ('om',): -0.33, ('vanavond', 'naar'): 0.4, ('ik',): -0.1})
#m.tweet_to_vector(['dit','even','uittesten','dan', 'maar'], {'dit': 1, 'even': 2, 'dacht':3, ('dit','even'):4})
