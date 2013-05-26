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
import nltk										# used for stemmer
from preprocessing import Preprocessing

class Main(object):
	# Read file
	DELIMITER = "\t"
	data = csv.reader(open("2000test_annotated.csv", 'rU'), delimiter=DELIMITER)

	# Dictionary for class, classify unkown into non-activity
	class_dict = {"Y": 0, "N": 1, "U": 2}

	# Distribution of trainingset, testset, validationset
	distribution = (0.7, 0.2, 0.1)

	testset = []
	trainset = []

	# Debug files
	DEBUG_SETS = "debug_sets.txt"

	def __init__(self, mode):
		""" Set up for training"""
		self.initialize()
		self.preprocess_tweets()
		self.create_sets()

	def initialize(self, mode):
		""" Initializes tweet and class sets """
		print "** Initialize.."
		for i, row in enumerate(self.data):
			# Ignores header
			if(i == 0):
				pass
			else:
			# TEMP, for testing only
				begin = 1000
				end = 1500
				if (i >= begin and i <= end):
					# TEMP, Only add if class is known! for testing only
					if (self.class_dict.get(row[5].upper()) is not None):
						# Get tweet and class 
						self.tweets[i-begin] = row[3]
						self.tweet_class[i-begin] = self.class_dict.get(row[5].upper())

	def preprocess_tweets(self, mode):
		""" Process tweets according to mode and set arrays """
		processObject = Preprocessing(mode, self.tweets)
		if ( "stem" in mode):
			self.stemmed_tweets_array = processObject.stemmed_tweets_array
		if ( "token" in mode):
			self.tokenized_tweets_array = processObject.tokenized_tweets_array
		if ( "pos" in mode): 
			self.pos_tweets_array = processObject.pos_tweets_array
		if ( "lemma" in mode)
			self.lemma_tweets_array = processObject.lemmatized_tweets_array

	def create_sets(self):
		""" Create training/test/validation set via indices """
		for i in range(0, len(self.tweets)):
			# Test if random number is smaller than distribution for trainset
			r_nr = random.random()
			if (r_nr < self.distribution[0]):
				self.trainSet.append(i)
			else:
				self.testSet.append(i)

		totallist = []
		totallist.append(self.trainSet)
		totallist.append(self.testSet)
		self.write_to_file(DEBUG_SETS, totallist

	def write_to_file(self, filename, array):
		"""	Dump array to file """
		f = file(filename, "w")
		pickle.dump(array, f)

	def read_from_file(self, filename, array):
		"""	Load array from file """
		f = file(filename, "r")
		array = pickle.load(f)

# call main with mode
m = Main("frog lemma pos stem token --debug")
