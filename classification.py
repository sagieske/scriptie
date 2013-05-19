import csv
import sys
import random
import nltk
import os
import operator

class Main():

	# Lists for tweet and class
	tweets = {}
	tweet_class = {}
	corpus = {}
	bigramcorpus = {}
	trigramcorpus = {}
	trainSet = []
	testSet = []

	# Dictionary for class, classify unkown into non-activity
	class_dict = {"Y": 0, "N": 1, "U": 2}

	# Distribution of trainingset, testset, validationset
	distribution = (0.7, 0.2, 0.1)

	# Read file
	DELIMITER = "\t"
	data = csv.reader(open("2000test_annotated.csv", 'rU'), delimiter=DELIMITER)

	def __init__(self):
		"""
		INIT
		"""
		# Load tweets from file
		self.initialize()
		self.count_classes()
		self.createSets()
		self.createCorpus("Frog")

	def initialize(self):
		"""
		Initializes tweet and class sets
		"""
		print "Initialize sets.."
		# Create tweet and class lists
		for i, row in enumerate(self.data):
			# Ignores header
			if(i != 0):
			# TEMP, for testing only
				if (i < 100):
					# Get tweet and class 
					self.tweets[i-1] = row[3]
					self.tweet_class[i-1] = self.class_dict.get(row[5].upper())
		print self.tweets[22]

	def createSets(self):
		"""
		Create training/test/validation set via indices 
		"""
		for i in range(1, len(self.tweets)):
			# Test if random number is smaller than distribution for trainset
			r_nr = random.random()
			if (r_nr < self.distribution[0]):
				self.trainSet.append(i)
			else:
				self.testSet.append(i)

	
	def count_classes(self):
		"""
		Counts and prints occurance of each class
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

	def createCorpus(self, mode):
		# USE FROG HERE? 
		# open frog to do all stemming of every sentence? sentences aan elkaar plakken, in frog gooien en weer uit elkaar halen?
		#if(mode == "Frog"):
		#	os.system("frog -t test.txt > frogtesting.txt")
		for index in self.tweets:
			tweet = self.tweets[index]	
			tweetclass = self.tweet_class[index]		
			
			# Split into tokens	
			tokens = nltk.word_tokenize(tweet)

			# add every token to corpus
			for index, item in enumerate(tokens):
				# check class
				if(tweetclass == 0):
					addition = (1,1)
				else:
					addition = (0,1)
				self.addToCorpus(tokens, index, addition, 1)


	def addToCorpus(self,tokens, index, addition, mode):
		"""
		adds items to corpus dependend of mode
		input: list of tokens, index, addition for dict, n-grams
		"""
		# Create ngrams
		if (index + mode <= len(tokens)):
			tupleItem = (tokens[index],)
			for i in range(index+1,index+mode):
					tupleItem = tupleItem + (tokens[i],)
			
			# Add ngrams in dictionary with addition
			if tupleItem in self.corpus:
				self.corpus[tupleItem] = tuple(map(operator.add, self.corpus[tupleItem], (addition)))
			else:
				self.corpus[tupleItem] = addition	

		print self.corpus


	def findHighest(self,corpus):
		value = 0
		best = {}
		for item in corpus:
			value1, value2 = corpus[item]
			if value1 > value:
				value = value1
				best = item
		print value
		print best

m = Main()
