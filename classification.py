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
from bagofwords import BagOfWords

class Main(object):
	"""
	Class for blablabla
	"""
	# Read file
	DELIMITER = "\t"
	data = csv.reader(open("2000test_annotated.csv", 'rU'), delimiter=DELIMITER)

	# Dictionary for class, classify unkown into non-activity
	class_dict = {"Y": 0, "N": 1, "U": 2}

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
	lemma_tweets_array = []

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
		self.print_sets()
		self.count_classes()

		# Dump sets
		if (self.dump):
			totallist = []
			totallist.append(self.trainset)
			totallist.append(self.testset)
			self.write_to_file(self.DEBUG_SETS, totallist)
		b = BagOfWords(self.stemmed_tweets_array, self.tweet_class, self.trainset)
		b.create_corpus(2)


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
				end = 1200
				if (i >= begin and i <= end):
					# TEMP, Only add if class is known! for testing only
					if (self.class_dict.get(row[5].upper()) is not None):
						# Get tweet and class 
						self.tweets[i-begin] = row[3]
						self.tweet_class[i-begin] = self.class_dict.get(row[5].upper())

		print self.tweet_class

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
			self.lemma_tweets_array = processObject.lemmatized_tweets_array

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


# call main with mode
m = Main("frog lemma pos stem token --debug")
