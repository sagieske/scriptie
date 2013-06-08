import nltk											# used for tokenization
import os											# used for calling Frog in new terminal
import operator
from pynlpl.clients.frogclient import FrogClient	# used for Frog client
import subprocess									# used for calling Frog in new terminal
import signal										# used for calling Frog in new terminal
import math	 
import nltk											# used for stemmer
import re											# used for regex
import time											# used for sleep
import pickle										# write and load list to file
import helpers

class Preprocessing(object):
	"""
	Class for preprocessing tweets. Tweets can be split into tokens in 4 different ways:
	- Tokenizing: tweets are split into tokens using nltk.tokenize
	- Stemming: tweets are split into stemmed tokens using Dutch Snowball stemmer
	- Lemmatized: tweets are split into lemmatized tokens (different verb tenses under same verb: e.g. 'drive', 'driving', 'drives' --> 
		'drive') using Frog, a Dutch morpho-syntactic analyzer and dependency parser
	- Part-Of-Speech: tweets are converted to POS tags and split into POS tokens using Frog, a Dutch morpho-syntactic analyzer and dependency 			parser

	Preprocessed arrays can be printed out.

	Parameters:
	----------

	mode : string
		Mode for how to tokenize tweets

	debug : boolean
		If True, program tries to read processed tokens from file.

	dump : boolean
		If True, program writes processed tokens to file

	tweets : dictionary of index key and string value
		Dictionary holds tweet strings and indexnumber

	lemmatized_tweets_array : array of arrays of strings
		Array of arrays of tweets split into lemmatized tokens

	pos_tweets_array : array of arrays of strings
		Array of arrays of tweets split into Part-Of-Speech tokens

	stemmed_tweets_array : array of arrays of strings
		Array of arrays of tweets split into stemmed tokens

	tokenized_tweets_array : array of arrays of strings
		Array of arrays of tweets split into tokens

	PORTNUMBER : int
		Portnumber for connecting with Frog server

	DEBUG_LEMMA : string
		Filename for writing and reading lemmatized tokens (for debugging)

	DEBUG_POS : string
		Filename for writing and reading Part-Of-Speech tokens (for debugging)

	DEBUG_STEM : string
		Filename for writing and reading stemmed tokens (for debugging)

	DEBUG_TOKEN : string
		Filename for writing and reading tokens (for debugging)


	"""

	DEBUG_STEM = "debug_stem.txt"
	DEBUG_TOKEN = "debug_token.txt"
	DEBUG_LEMMA = "debug_lemma.txt"
	DEBUG_POS = "debug_pos.txt"

	PORTNUMBER = 1160

	tweets = {}
	stemmed_tweets_array = []
	tokenized_tweets_array = []
	lemmatized_tweets_array = []
	pos_tweets_array = []

	def __init__(self, mode, tweetlist):
		"""	Initialize tweets for use"""
		self.tweets = tweetlist
  		self.debug = "--debug" in mode
  		self.dump = "--write" in mode
		self.mode = re.sub(r' --(\S+)', r'', mode)


		#self.preprocess_tweets()

	def preprocess_tweets(self):
		"""	Call functions to process tweets according to mode"""
		print "** Preprocessing with MODE: %s\n   DEBUG: %s\n   WRITE TO FILE: %s" % (self.mode, self.debug, self.dump)
		mode_args = self.mode.split()
		if("stem" in mode_args):
			self.stemming()
		if("token" in mode_args):
			self.tokenize()
		if("frog" in mode_args):
			self.frogtokens()

		if ( self.dump ):
			print "DUMPING TO FILE"
			self.write_all_to_file()
				
	def stemming_str(self):
		""" Print out stemmed_tweets_array as string"""
		print ">> Stemmed:"
		for item in self.stemmed_tweets_array:
			print '- ' + '|'.join(item)

	def tokenize_str(self):
		""" Print out tokenized_tweets_array as string"""
		print ">> Tokenized:"
		for item in self.tokenized_tweets_array:
			print '- ' + '|'.join(item)

	def lemmatized_str(self):
		""" Print out lemmatized_tweets_array as string"""
		print ">> Lemmatized:"
		for item in self.lemmatized_tweets_array:
			print '- ' + '|'.join(item)

	def pos_str(self):
		""" Print out pos_tweets_array as string"""
		print ">> POS tagged:"
		for item in self.pos_tweets_array:
			print '- ' + '|'.join(item)

	def stemming(self):
		""" Stem all tweets given to object and set to array """
		print "** Stemming.."
		debug = self.debug
		if (debug):
			try:
				self.read_from_file(self.DEBUG_STEM, "stem")
			except:
				print "! Error in reading from file debug.txt. Redo stemming"
				debug = False
		if (not debug):
			for index in self.tweets:
				# Snowball stemmer has problem with c, delete characters
				tweet_ascii = filter(lambda x: ord(x) < 128, self.tweets[index])
				stemmed_tweet = self.stem_tweet(tweet_ascii)		
				self.stemmed_tweets_array.append(stemmed_tweet)

	def stem_tweet(self, tweet):
		""" Stem tweet string and return array of stemmed words """
		stemmer = nltk.stem.SnowballStemmer('dutch')
		stemmed_tweet = stemmer.stem(tweet)
		stemmed_tweet = stemmed_tweet.split()
		return stemmed_tweet
		
	def tokenize(self):
		""" Tokenize all tweets given to object and set to array """
		print "** Create tokens.."
		debug = self.debug
		if (debug):
			try:
				self.read_from_file(self.DEBUG_TOKEN, "token")
			except:
				print "! Error in reading from file debug.txt. Redo tokenization"
				debug = False
		if (not debug):
			for index in self.tweets:
				tokens = nltk.word_tokenize(self.tweets[index])
				self.tokenized_tweets_array.append(tokens)

	def frogtokens(self):
		""" Lematize all tweets given to object using frog server/client and set to array """
		debug = self.debug
		if (debug):
			try:
				if( "lemma" in self.mode ):
					self.read_from_file(self.DEBUG_LEMMA, "lemma")
				if( "pos" in self.mode ):
					self.read_from_file(self.DEBUG_POS, "pos")
			except:
				print "! Error in reading from file debug.txt. Redo frogtokens"
				debug = False
		if(not debug):
			self.startFrogServer('start')			
			time.sleep(15)							# Time for startup server
			frogclient = FrogClient('localhost',self.PORTNUMBER)
			if( "lemma" in self.mode ):
				print "** Creating lemma's.."

			if ( "pos" in self.mode ):
				print "** Creating POS tags.."
			
			# Get frog analyses
			for index in self.tweets:
				tokensword, tokenslemma, tokenspos = self.frog_tweets(frogclient, self.tweets[index])
				if ( tokenslemma ):
					self.lemmatized_tweets_array.append(tokenslemma)
				if ( tokenspos ):
					self.pos_tweets_array.append(tokenspos)
			self.startFrogServer('stop')

	def startFrogServer(self, mode):
		""" Starts/stops Frog server in seperate terminal """
		if(mode == 'start'):
			print "** Start Frog Server"
			os.system("mate-terminal -e 'frog -S " + str(self.PORTNUMBER) + "'")
		if(mode == 'stop'):
			print "** Close Frog Server"
			proc = subprocess.Popen(["pgrep", 'frog'], stdout=subprocess.PIPE) 
			for pid in proc.stdout: 
				os.kill(int(pid), signal.SIGTERM)

	def frog_tweets(self, frogclient, tweet):
		"""	Use frog for processing according to mode. Return array of processed words """
		frogtweet = frogclient.process(tweet.lower())
		tokensword = []
		tokenslemma = []
		tokenspos = []
		for test in frogtweet:
			# Frog sometimes contains tuple of None
			if (None in test):
				pass
			else:
				word, lemma, morph, pos = test
				if('word' in self.mode):
					tokensword.append(word)
				if('lemma' in self.mode):
					tokenslemma.append(lemma)
				if('pos' in self.mode):
					tokenspos.append(pos)
		return (tokensword, tokenslemma, tokenspos)

	def read_from_file(self, filename, frogmode):
		"""	Load array from file """
		f = file(filename, "r")

		# Load into correct array
		if ( "stem" in frogmode):
			self.stemmed_tweets_array = pickle.load(f)
		if ( "token" in frogmode):
			self.tokenized_tweets_array = pickle.load(f)
		if ( "pos" in frogmode): 
			self.pos_tweets_array = pickle.load(f)
		if ( "lemma" in frogmode):
			self.lemmatized_tweets_array = pickle.load(f)

	def write_all_to_file(self):
		"""	Dumps all filled arrays to file """
		print "WRITE TO FILE"
		if ( self.stemmed_tweets_array ):
			helpers.dump_to_file(self.DEBUG_STEM, self.stemmed_tweets_array)
		if ( self.tokenized_tweets_array ):
			helpers.dump_to_file(self.DEBUG_TOKEN, self.tokenized_tweets_array)
		if ( self.lemmatized_tweets_array ):
			helpers.dump_to_file(self.DEBUG_LEMMA, self.lemmatized_tweets_array)
		if ( self.pos_tweets_array ):
			helpers.dump_to_file(self.DEBUG_POS, self.pos_tweets_array)
