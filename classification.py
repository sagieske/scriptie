import csv
import sys
import random
import nltk
import os
import operator
import time
import socket 
from pynlpl.clients.frogclient import FrogClient


class Main():

	# Lists for tweet and class
	tweets = {}
	tweet_class = {}
	corpus = {}
	corpus_weights = {}
	bigramcorpus = {}
	trigramcorpus = {}
	trainSet = []
	testSet = []

	totalPos = 0
	totalNeg = 0

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
		self.readFrog()
		#tweet = "eens even kijken hoe ik hier naar kijk bla &"
 		#self.processTokens(tweet,"lemma")

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
				if (i > 1000 and i < 1500):
					# Get tweet and class 
					self.tweets[i-1000] = row[3]
					self.tweet_class[i-1000] = self.class_dict.get(row[5].upper())
		frogclient = FrogClient('localhost',1126)
		for item in frogclient.process("Laten we kijken hoe we dit kunnen uittesten"):
			print item


	def printTweetsToText(self):
		print "writing"
		f = open('testcode.txt','w')
		for index in self.tweets:
			if(index < 100):
				f.write(self.tweets[index]+ '\n')
			else:
				break
		f.close()

	def useFrog(self):
		now= time.time()
		print "START frog"
		os.system("frog -n -t testcode.txt -o frogtesting.txt")
		print "DONE frog"
		timetaken = time.time() - now
		print "time taken: %d" %timetaken

	def readFrog(self):
		fo = open("frogtesting.txt", "r")
		tokens = []
		for line in fo:
			if line != "\n":
				newline = line.split("\t")
				tokens.append(newline)
			else:
				break
		fo.close()
		print tokens

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
					addition = (1,0)
					self.totalPos += 1
				else:
					addition = (0,1)
					self.totalNeg += 1
				self.addToCorpus(tokens, index, addition, 2)

	def processTokens(self, tweet,mode):
		if mode == "tk":
			tokens = nltk.word_tokenize(tweet)
		if mode == "lemma":
			# TODO: VIA PORT is sneller ws
			tokens = []
			# print line to file
			f = open('testcode.txt','w')
			f.write(tweet)
			f.close()
			time.sleep(3)
			# use frog
			os.system("frog -t testcode.txt > frogtesting.txt")
			# read from frog file
			fo = open("frogtesting.txt", "r")
			for line in fo:
				if line != "\n":
					newline = line.split("\t")
					print newline[2]
					tokens.append(newline[2])
		return tokens

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
		self.findHighest(self.corpus_weights, 10)
		self.findLowest(self.corpus_weights, 10)

	def findHighest(self,corpus, nr):
		"""
		Print out max <nr> of corpus
		"""
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=True)[:nr])
		print topCorpus

	def findLowest(self,corpus, nr):
		"""
		Print out min <nr> of corpus
		"""
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=False)[:nr])
		print topCorpus

m = Main()
