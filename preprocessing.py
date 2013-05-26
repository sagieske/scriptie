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

class Preprocessing(object):
	"""
	Class for preprocessing tweets
	blablaballa etc

	"""
	portnumber = 1150	# Portnumber for Frog

	stemmed_tweets_array = []
	tokenized_tweets_array = []
	lemmatized_tweets_array = []

	def __init__(self, mode, tweetarray):
		"""	Initialize tweetarray for use"""
		self.tweetarray = tweetarray
		self.mode = mode

	def stemming_str(self):
		for item in self.stemmed_tweets_array:
			print '|'.join(item)

	def tokenize_str(self):
		for item in self.tokenized_tweets_array:
			print '|'.join(item)

	def lemmatization_str(self):
		for item in self.lemmatized_tweets_array:
			print '|'.join(item)

	def stemming(self):
		""" Stem all tweets given to object and set to array """
		for item in self.tweetarray:
			stemmed_tweet = self.stem_tweet(item)		
			self.stemmed_tweets_array.append(stemmed_tweet)

	def stem_tweet(self, tweet):
		""" Stem tweet string and return array of stemmed words """
		stemmer = nltk.stem.SnowballStemmer('dutch')
		stemmed_tweet = stemmer.stem(tweet)
		stemmed_tweet = stemmed_tweet.split()
		return stemmed_tweet
		
	def tokenize(self):
		""" Tokenize all tweets given to object and set to array """
		for item in self.tweetarray:
			tokens = nltk.word_tokenize(item)
			self.tokenized_tweets_array.append(tokens)

	def lemmmatization(self, frogmode):
		""" Lematize all tweets given to object using frog server/client and set to array """
		self.startFrogServer('start')			
		time.sleep(15)							# Time for startup server
		frogclient = FrogClient('localhost',self.portnumber)

		for item in self.tweetarray:
			tokens = self.frog_tweets(frogclient, item, frogmode)
			# Write tokens to file for later testing
			self.lemmatized_tweets_array.append(tokens)

		self.startFrogServer('stop')

	def startFrogServer(self, mode):
		""" Starts/stops Frog server in seperate terminal """
		if(mode == 'start'):
			print "Start Frog Server"
			os.system("mate-terminal -e 'frog -S " + str(self.portnumber) + "'")
		if(mode == 'stop'):
			print "Close Frog Server"
			proc = subprocess.Popen(["pgrep", 'frog'], stdout=subprocess.PIPE) 
			for pid in proc.stdout: 
				os.kill(int(pid), signal.SIGTERM)

	def frog_tweets(self, frogclient, tweet, frogmode):
		"""	Use frog for processing according to mode. Return array of processed words """
		frogtweet = frogclient.process(tweet.lower())
		tokens = []
		for test in frogtweet:
			# Frog sometimes contains tuple of None
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
		return tokens
