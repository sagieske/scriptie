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
import nltk.stem as Stemmer							# used for stemmer

"""
TODO:
- combinations of unigram/POStag etc
- easy way to not start up Frog everytime for testing
"""


class Main(object):

	# Lists for tweet and class
	tweets = {}
	tweet_class = {}
	corpus = {}
	corpus_weights = {}
	bigramcorpus = {}
	trigramcorpus = {}
	trainSet = []
	testSet = []
	bagOfWords = []

	totalPos = 0
	totalNeg = 0
	
	# Portnumber for Frog
	portnumber = 1150

	# Dictionary for class, classify unkown into non-activity
	class_dict = {"Y": 0, "N": 1, "U": 2}

	# Distribution of trainingset, testset, validationset
	distribution = (0.7, 0.2, 0.1)

	# Read file
	DELIMITER = "\t"
	data = csv.reader(open("2000test_annotated.csv", 'rU'), delimiter=DELIMITER)

	def __init__(self, mode):
		"""
		INIT
		"""
		self.initialize(mode)

		self.count_classes()
		self.createCorpus(mode)
		self.setCorpusWeights()
		self.scaleCorpusWeights()

		self.bagOfWords = self.corpus_weights.keys()


	def initialize(self, mode):
		"""
		Initializes tweet and class sets
		"""
		print "Initialize.."


		# Create tweet and class lists
		for i, row in enumerate(self.data):
			# Ignores header
			if(i != 0):
			# TEMP, for testing only
				begin = 1000
				end = 1500
				if (i >= begin and i <= end):
					# Only add if class is known! for testing
					if (self.class_dict.get(row[5].upper()) is not None):
						# Get tweet and class 
						self.tweets[i-begin] = row[3]
						self.tweet_class[i-begin] = self.class_dict.get(row[5].upper())

		modelist = mode.split()
		# Create Train and Test
		if (modelist[0] == "frog"):
			self.createSets()

		# DEBUG: read sets from file
		if (modelist[0] == "frogdb"):
			f = file("sets.txt", "r")
			totallist = pickle.load(f)
			self.trainSet = totallist[0]
			self.testSet = totallist[1]
			print self.testSet



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
		
		print self.testSet

		# DEBUG: write to file
		totallist = []
		totallist.append(self.trainSet)
		totallist.append(self.testSet)
		f = file("sets.txt", "w")
		pickle.dump(totallist, f)
	
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
		modelist = mode.split()

		if(modelist[0] == "frogdb"):
			#WARNING TRAIN AND TEST SET ARE DIFFERENT, IS NOT WRITEN TO FILE! 
			frogclient = None
			tokenarray = []
			for line in open('frog_tokens.txt','r'):
				tokens = line.split("\t")
				tokenarray.append(tokens)
			print len(tokenarray)
			for tokens in tokenarray:
				tweetclass = int(tokens[len(tokens)-1])
				del tokens[-1]
				# add every token to corpus
				self.addTokensToCorpus(tokens, tweetclass, 2)

					
		# Start Frog server and client
		if(modelist[0] == "frog"):
			self.startFrogServer('start')
			time.sleep(15)
			frogclient = FrogClient('localhost',self.portnumber)
			open("frog_tokens.txt", 'w').close()

			print "Testing trainset size:"
			print len(self.trainSet)

			for index in self.trainSet:
				tweet = self.tweets[index]	
				tweetclass = self.tweet_class[index]		
			
				# Split into tokens	
				tokens = self.createTokens(frogclient, index, mode)

				self.addTokensToCorpus(tokens, tweetclass, 2)

		if(modelist[0] == "Frog"):
			# Stop Server
			self.startFrogServer('stop')

	def addTokensToCorpus(self,tokens,tweetclass, ngram):
		"""
		add every token to corpus
		"""
		for index, item in enumerate(tokens):
			if(tweetclass == 0):
				addition = (1,0)
				self.totalPos += 1
			else:
				addition = (0,1)
				self.totalNeg += 1
			self.addToCorpus(tokens, index, addition, ngram)

	def createTokens(self,frogclient, index,mode):
		"""
		Creates tokens for corpus dependent of mode
		"""
		tokens = []
		modelist = mode.split()
		if modelist[0] == "tk":
			print "tk"
			tokens = nltk.word_tokenize(tweet)

		if modelist[0] == "frogdebug":
			pass
		if modelist[0] == "frog":
			tweet = self.tweets[index]
			frogtweet = frogclient.process(tweet.lower())
			tokens = self.processFrogtweet(tweet, frogtweet, modelist[1])
			# Write tokens to file for later testing
			with open("frog_tokens.txt", "a") as myfile:
				stringtokens = self.tokensToString(tokens)
				myfile.write(stringtokens.encode('utf-8') + str(self.tweet_class[index]).encode('utf-8') + "\n".encode('utf-8'))
		return tokens

	def tokensToString(self,tokens):
		string = ""
		for item in tokens:
			string += item
			string += "\t"
		return string
		
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
		"""
		Starts/stops Frog server in seperate terminal
		"""
		if(mode == 'start'):
			print "start"
			os.system("mate-terminal -e 'frog -S " + str(self.portnumber) + "'")
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


	def scaleCorpusWeights(self):
		"""
		Scale weights to [-1,1]
		"""
		# Get old range
		oldMax = max(self.corpus_weights.iteritems(), key=operator.itemgetter(1))[1]
		oldMin = min(self.corpus_weights.iteritems(), key=operator.itemgetter(1))[1]
		print "scaling from [%f,%f] to [-1,1]" %(float(oldMax),float(oldMin))
		# Set new range
		newMin = float(-1.0)
		newMax = float(1.0)
		
		# Calculate new values
		print "Before"
		print len(self.corpus_weights)
		delkeys = []
		for key in self.corpus_weights:
			oldValue = float(self.corpus_weights[key])
			newValue = (((oldValue - oldMin) * (newMax - newMin)) / (oldMax - oldMin)) + newMin
			if(newValue < 0.25 and newValue > -0.25):
				delkeys.append(key)
			else:	
				self.corpus_weights[key] = newValue

		for key in delkeys:
			del self.corpus_weights[key]
		print "After"
		print len(self.corpus_weights)
		self.findHighest(self.corpus_weights, 10)
		self.findLowest(self.corpus_weights, 10)
		

	def findHighest(self,corpus, nr):
		"""
		Print out max <nr> of corpus
		"""
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=True)[:nr])
		for item in topCorpus:
			value = topCorpus[item]
			print "(%s, %s) : %f" % (item[0], item[1], value)
		#print topCorpus

	def findLowest(self,corpus, nr):
		"""
		Print out min <nr> of corpus
		"""
		topCorpus = dict(sorted(corpus.iteritems(), key=operator.itemgetter(1), reverse=False)[:nr])
		for item in topCorpus:
			value = topCorpus[item]
			print "(%s, %s) : %f" % (item[0], item[1], value)
		#print topCorpus
"""
# The docstring for a class should summarize its behavior and list the public methods and instance variables
# preproccessing: process all tweets in particular way, several functions for severall preprocessing
# input: Mode of preproccessing, tweets array
"""

class Preprocessing(object):
	"""
	Class for preprocessing tweets
	blablaballa etc

	"""
	stemmed_tweets_array = []

	def __init__(self, mode, tweetarray):
		"""	Initialize tweetarray for use"""
		self.tweetarray = tweetarray
		self.mode = mode

	def stemming_str(self):
		for item in self.stemmed_tweets_array:
			print item

	def stemming(self):
		""" Stem all tweets given to object and set to array """
		for item in self.tweetarray:
			stemmed_tweet = self.stem_tweet(item)		
			self.stemmed_tweets_array.append(stemmed_tweet)

	def stem_tweet(self, tweet):
		""" Stem tweet string and return array of stemmed words """
		stemmer = Stemmer.SnowballStemmer('dutch')
		stemmed_tweet = stemmer.stem(tweet)
		stemmed_tweet = stemmed_tweet.split()
		return stemmed_tweet
		
test = Preprocessing("test", ["Dit is een tweet die uitgeprobeerd gaat worden", "test dit testte even kijken"])
test.stemming()
test.stemming_str()

#m = Main("frog lemma")
#m = Main("frogdb lemma")
