import csv
import sys
import random
import nltk
import os
import operator
import time
import socket 
from pynlpl.clients.frogclient import FrogClient
import subprocess
import signal
import math

"""
TODO:
- combinations of unigram/POStag etc
- easy way to not start up Frog everytime for testing
"""


class Main():

	# Lists for tweet and class
	tweets = {}
	tweet_class = {}
	corpus = {}
	corpus_weights = {}
	corpus_weights_test = {}
	bigramcorpus = {}
	trigramcorpus = {}
	trainSet = []
	testSet = []

	totalPos = 0
	totalNeg = 0
	
	# Portnumber for Frog
	portnumber = 1200

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
		self.setCorpusWeights()
		#self.printTweetsToText()
		#self.useFrog()
		#self.readFrog()
		#tweet = "eens even kijken hoe ik hier naar kijk bla &"
 		#self.createTokens('ik kijkte naar gekeken materiaal en brave braaf zwetende zweet bah! ?',"frog lemma")

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
				if (i > 1000 and i < 1400):
					# Only add if class is known! for testing
					if (self.class_dict.get(row[5].upper()) is not None):
						# Get tweet and class 
						self.tweets[i-1000] = row[3]
						self.tweet_class[i-1000] = self.class_dict.get(row[5].upper())

	def createSets(self):
		"""
		Create training/test/validation set via indices 
		"""
		for i in range(0, len(self.tweets)):
			# Test if random number is smaller than distribution for trainset
			r_nr = random.random()
			if (r_nr < self.distribution[0]):
				self.trainSet.append(i)
			else:
				self.testSet.append(i)

	
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

	def createCorpus(self, mode):
		"""
		Create training corpus
		"""
		if(mode == "Frog"):
			# Start Frog Server
			self.startFrogServer('start')
			# Wait for Frog to startup
			time.sleep(15)
			# Start Frog Client
			frogclient = FrogClient('localhost',self.portnumber)

		# USE FROG HERE? 
		# open frog to do all stemming of every sentence? sentences aan elkaar plakken, in frog gooien en weer uit elkaar halen?
		#if(mode == "Frog"):
		#	os.system("frog -t test.txt > frogtesting.txt")
		for index in self.tweets:
			tweet = self.tweets[index]	
			tweetclass = self.tweet_class[index]		
			
			# Split into tokens	
			tokens = self.createTokens(frogclient, tweet,"frog word")

			# add every token to corpus
			for index, item in enumerate(tokens):
				# check class
				if(tweetclass == 0):
					addition = (1,0)
					self.totalPos += 1
				else:
					addition = (0,1)
					self.totalNeg += 1
				self.addToCorpus(tokens, index, addition, 3)

		if(mode == "Frog"):
			# Stop Server
			self.startFrogServer('stop')

	def createTokens(self,frogclient, tweet,mode):
		"""
		Creates tokens for corpus dependent of mode
		"""
		tokens = []
		modelist = mode.split()
		if modelist[0] == "tk":
			print "tk"
			tokens = nltk.word_tokenize(tweet)
		if modelist[0] == "frog":
			#print "frog"
			# Process tweet
	 		# TODO: test is lower
			frogtweet = frogclient.process(tweet.lower())
			tokens = self.processFrogtweet(tweet.lower(), frogtweet, modelist[1])

		return tokens

	def processFrogtweet(self, tweet, frogtweet, frogmode):	
		"""
		Process Frog information for requested items into token list
		""" 
		tokens = []

		for test in frogtweet:
			# frog sometimes contains tuple of None
			if (None in test):
				pass
			else:
				word, lemma, morph, pos = test
				if(frogmode == 'word'):
					tokens.append(word)
				if(frogmode == 'lemma'):
						tokens.append(lemma)
				if(frogmode == 'pos'):
					tokens.append(pos)
				if(frogmode == 'wordpos'):
					token = (word, pos)
					tokens.append(token)
		return tokens
		

	def startFrogServer(self, mode):
		if(mode == 'start'):
			print "start"
			os.system("mate-terminal -e 'frog -S 1200'")
		if(mode == 'stop'):
			print "stop"
			proc = subprocess.Popen(["pgrep", 'frog'], stdout=subprocess.PIPE) 
			for pid in proc.stdout: 
				os.kill(int(pid), signal.SIGTERM)


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

	def setCorpusWeights(self):
		"""
		Set weights for words. Remove singular occurances.
		"""
		counter = 0
		for key,value in self.corpus.iteritems():
			value_pos, value_neg = value

			# Token occures more than once
			if (sum(value) >1):				
				# Calculate positive and negative influence
				positive = 0
				negative = 0
				if (value_pos > 0):
					positive = value_pos/float(self.totalPos)
				if (value_neg > 0):
					negative = value_neg/float(self.totalNeg)
				valueweight = positive - negative
				# Set value
				if (valueweight != 0):
					self.corpus_weights[key] = valueweight

				else:
					counter += 1
		self.scaleCorpusWeights()


	def scaleCorpusWeights(self):
		print "scaling"

		oldMax = max(self.corpus_weights.iteritems(), key=operator.itemgetter(1))[1]
		oldMin = min(self.corpus_weights.iteritems(), key=operator.itemgetter(1))[1]
		print "scaling from [%f,%f] to [-1,1]" %(float(oldMax),float(oldMin))
		newMin = float(-1.0)
		newMax = float(1.0)
		
		for key in self.corpus_weights:
			oldValue = float(self.corpus_weights[key])
			newValue = (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin
			self.corpus_weights[key] = newValue
			if newValue < 0:
				print newValue

		self.findHighest(self.corpus_weights, 10)
		self.findLowest(self.corpus_weights, 10)		
		self.createNewCorpusWeights()

	def createNewCorpusWeights(self):
		counter =0
		counter5 =0
		counter6 = 0
		for key in self.corpus_weights:
			value =  self.corpus_weights[key]
			if (value > 0.5 or value < -0.5):
				self.corpus_weights_test[key] = self.corpus_weights[key]

		print len(self.corpus_weights)
		print len(self.corpus_weights_test)

		#self.corpus_weights.update({n: 2 * my_dict[n] for n in my_dict.keys()})

	def findHighest(self,corpus, nr):
		"""
		Print out max <nr> of corpus
		"""
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=True)[:nr])
		for item in topCorpus:
			value = topCorpus[item]
			print "(%s, %s, %s) : %f" % (item[0], item[1], item[2], value)
		#print topCorpus

	def findLowest(self,corpus, nr):
		"""
		Print out min <nr> of corpus
		"""
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=False)[:nr])
		for item in topCorpus:
			value = topCorpus[item]
			print "(%s, %s, %s) : %f" % (item[0], item[1], item[2], value)
		#print topCorpus

m = Main()
