import csv											# used for import csv tweets
import os											# used for calling Frog in new terminal
import operator
import time											# used for timer
from pynlpl.clients.frogclient import FrogClient	# used for Frog client
import helpers
import pickle
import re
from collections import defaultdict
import subprocess									# used for calling Frog in new terminal
import signal										# used for calling Frog in new terminal
import logging, gensim, bz2
import gensim										# used for LDA
import nltk
import unicodedata



class TopicExtraction(object):
	"""

	"""
	# Read file
	DELIMITER = "\t"
	TOPICFILE = "2000test_annotated_v3_class.csv"
	class_dict = {"Y": 0, "N": 1, "U": 1}

	short_tweets = []
	tweets = []
	tweettokens = []
	posdeleted_tweets = []
	tweetpos_tokens = []
	tuples = []
	short_tweets = []
	short_pos = []
	tweetpos_tokenstypes = []
	dictionary_pos = {}
	lemmatweets = []
	PORTNUMBER = 1122

	EMOTICONS = ['[=:]-*[()DdPpSsOo(\|)(\$)]']
	EMOTICONS_2 = ['\(.\)']
	NUMBERS = ['\d+[(\.):]\d+', '\d+']

	def __init__(self, debug):
		""" Initializes tweet and class sets """
		print "** Initialize.."
		self.debug = debug
		self.load_stopword_file()
		data = csv.reader(open(self.TOPICFILE, 'rU'), delimiter=self.DELIMITER)
		for i, row in enumerate(data):
			if i == 0:
				pass
			if (row[7] == '0'):
				tweet_enc = self.remove_accents(row[3])
				substituted = re.sub(r'\.\.+\s', r' ', tweet_enc)
				substituted = self.remove_stopwords(substituted)
				self.tweets.append(substituted)
		self.dictionary_pos = defaultdict(list)

	def remove_accents(self,datainput):
		""" Removes accents from characters """
		data = datainput.decode('unicode-escape')
		return''.join(c for c in unicodedata.normalize('NFD', data) if unicodedata.category(c) != 'Mn')


	def create_wordpostuples(self, array):
		""" Create tokens and POS tags for tweets """
		filename = self.TOPICFILE.split()[0]
		wordpos_filename = filename + "_wordpos.txt"

		readfromfile = self.debug
		if (readfromfile):
			try:
				self.tuples = helpers.read_from_file(wordpos_filename)
			except: 
				print "! Error in reading from file. Redo posword tuples"
				readfromfile = False

		if (not readfromfile):
			self.startFrogServer('start')			
			time.sleep(20)							# Time for startup server
			frogclient = FrogClient('localhost',self.PORTNUMBER)
			print "** START frog analysis."
			print "** Creating POS tags.. (This may take a while)"
			for item in array:
				lemmapos_array = self.frog_tweets(frogclient, item)
				self.tuples.append(lemmapos_array)	
		
			helpers.dump_to_file(wordpos_filename, self.tuples)
			self.startFrogServer('stop')			
	

	def startFrogServer(self, modus):
		""" Starts/stops Frog server in seperate terminal """
		if(modus == 'start'):
			print "** Start Frog Server"
			os.system("frog -S " + str(self.PORTNUMBER) + " > /dev/null 2>&1 &")
			#os.system("mate-terminal -e 'frog -S " + str(self.PORTNUMBER) + "'")
		if(modus == 'stop'):
			print "** Close Frog Server"
			proc = subprocess.Popen(["pgrep", 'frog'], stdout=subprocess.PIPE) 
			for pid in proc.stdout: 
				os.kill(int(pid), signal.SIGTERM)

	def frog_tweets(self, frogclient, tweet):
		"""	Use frog for processing according to mode. Return array of processed words """
		frogtweet = frogclient.process(tweet.lower())
		tuples_tweet = []
		for worditem in frogtweet:
			# Frog sometimes contains tuple of None
			if (None in worditem):
				pass
			else:

				word, lemma, morph, pos = worditem
				if 'vanavond' in word:
					pos = 'BW(vanavond)'
				deleted = self.deletepos(word,pos)
				if (deleted):
					pass
				else:
					tuple_pos = (word,pos)
					tuples_tweet.append(tuple_pos)
					#lemma_tweet.append(lemma)

		return tuples_tweet
	

	def begin_ngram_dictionary(self, ngramsize):
		""" Update dictionary of POS tags with ngramsize
		"""
		for item in self.tuples:
			self.ngram_pos(item, ngramsize)

	def ngram_pos(self, tuple_array, ngramsize):
		""" Create POStag tuples and update dictionary
		"""
		for index in range(0, len(tuple_array)-ngramsize+1):
			# Create tuples
			tupleItem_words = (tuple_array[index][0],)
			tupleItem_pos = (tuple_array[index][1],)
			for i in range(index+1,index+ngramsize):
				tupleItem_words = tupleItem_words + (tuple_array[i][0],)
				tupleItem_pos = tupleItem_pos + (tuple_array[i][1],)

			# Update dictionary
			if tupleItem_pos in self.dictionary_pos:
				value = self.dictionary_pos[tupleItem_pos]
				value.append(tupleItem_words)
				newvalue = list(set(value))
				self.dictionary_pos[tupleItem_pos] = newvalue
			else:
				self.dictionary_pos[tupleItem_pos] = [tupleItem_words]

	def create_pos_corpus(self, ngramsize, array):
		""" Create training corpus """
		corpus = {}
		for index, tuple_tweet in enumerate(array):
			corpus = self.add_tokens_to_corpus(tuple_tweet, ngramsize, corpus)

		deletekeys = []
		for corpusindex, tuplekey in enumerate(corpus):
			for tuplekeyindex,item in enumerate(tuplekey):
				if (tuplekeyindex < len(tuplekey)-1):
					if tuplekey[tuplekeyindex] == tuplekey[tuplekeyindex+1]:
						deletekeys.append(tuplekey)
						#print "DEL"
						#print tuplekey

		for item in deletekeys:
			corpus.pop(item, None)

		corpus.update((x, y*ngramsize*ngramsize) for x, y in corpus.items())

		return corpus



	def add_tokens_to_corpus(self,tokens, ngramsize, corpus):
		""" add every token to corpus"""
		for index in range(0, len(tokens)-ngramsize+1):
			# Create tuple
			tupleItem = (tokens[index],)
			
			for i in range(index+1,index+ngramsize):
				tupleItem = tupleItem + (tokens[i],)
			
			# Add ngrams in dictionary with addition
			corpus[tupleItem] = corpus.get(tupleItem, 0) +1
		return corpus

	def print_out_version(self, tweet, postweet):

		flat_postweet = " ".join(postweet)
		sortedarray = sorted(self.corpus.iteritems(), key=operator.itemgetter(1), reverse=True)
		# Ga alle opties af, van best naar slechts
		for (item, freq) in sortedarray:
			# Flatten tuple
			flat_tuple = " ".join(list(item))
			# Check if tuple in postweet, If so create short tweet and return
			if flat_tuple in flat_postweet:
				splittweet = tweet.split()
				# Get tweet only consisting of this pos.
				startindex = self.get_startindex_sublist(list(item), postweet)
				print " ".join(splittweet[startindex:startindex+len(item)])
				break

	def get_startindex_sublist(self, sublist, wholelist):
		""" Get start index of sublist in wholelist and return. No index found then return -1 """		
		startindex = -1
		for index, item in enumerate(wholelist):
			if item == sublist[0]:
				if sublist == wholelist[index:index+len(sublist)]:
					startindex = index
		return startindex

	def tweetparts(self):
		""" Get substrings of tweet containing vanavond """
		for index, item in enumerate(self.tweettokens):
			newitem = " ".join(item)
			(stringpart, pospart) = self.get_tweetpart('vanavond',  newitem, self.tweetpos_tokens[index])
			self.short_tweets.append(stringpart)
			self.short_pos.append(pospart)



	def get_unzippedtuples_array(self, tuplearray):
		""" Create seperate arrays from tuples"""
		for tweet_tuples in tuplearray:
			tupletweet = []
			tuplepos = []

			unzipped_tuples = [list(t) for t in zip(*tweet_tuples)]
			self.tweettokens.append(unzipped_tuples[0])
			self.tweetpos_tokens.append(unzipped_tuples[1])


	def deletepos(self, word, pos):
		""" Tests if POS tag belongs to list to be removed. Returns Boolean"""
		# POS tags to be removed	
		#poslist_delete = ['TW(rang', 'TSW', 'BW', 'VNW', 'WW(vd', 'LID', 'VG', 'ADJ(nom', 'ADJ(dial)', 'comp', 'SPEC', 'WW(od', 'sup', 'ADJ(postnom']	
		# VG: voegwoorden, TW: rang telwoorden, ADJ:adjectief, TSW: tussenwerpsel, LID: lidwoord, 	VNW: voornaamwoord
		# prenom: placed before noun, sup: superlatief, hoogte graag van verbuiging, comp: comperatief
		poslist_delete = ['TW(rang', 'ADJ(prenom, sup', 'TSW', 'BW', 'LID', 'VG', 'comp', 'VNW(onbep']
		poslist_notdelete = ['VNW(onbep,pron', 'BW(vanavond)']		# words like 'niets, niks'		
		#poslist = []
		for item in poslist_delete:
			if item in str(pos):
				if 'vanavond' in word:		# do not delete vanavond (BW())
					return False
				else:
					for item in poslist_notdelete:
						if item in str(pos):
							return False
					return True
		return False
	
	def remove_stopwords(self,inputsentence):
		""" Removes stop words in sentences. Returns substituted sentence """

		wordlist = []
		#pattern_time = re.compile('vanavond|morgen|vandaag|vanmiddag|gister|gisteren|eerst|daarna')
		#sentence = pattern_time.sub('', sentence)

		# Delete 'links'
		r_link1 = re.compile(r"(http:)[^ ]*")
		sentence = r_link1.sub(' ', inputsentence)
		r_link2 = re.compile(r"(www.)[^ ]*")
		sentence = r_link2.sub(' ', sentence)
		
		# Remove characters which occur more than twice immediately after eachother
		expr = r'(.)\1{3,}'
		replace_by = r'\1\1'
		sentence = re.sub(expr, replace_by, sentence)


		# append words like z'n 
		sentence = re.sub("'", "", sentence)
		# substitute _

		pattern_nonword = re.compile(r'[^\w\.\s|\,|\?|!|;|_]+')
		sentence = pattern_nonword.sub(" ", sentence) 


		#self.STOPWORD_FILE +
		wordlist +=  self.STOPWORD_FILE + self.NUMBERS + self.EMOTICONS + self.EMOTICONS_2 + ['[Rr][Tt]']
		for x in wordlist:
			sentence = re.sub(' '+x+' ',' ', sentence)
			sentence = re.sub('\A'+x+' ',' ', sentence)
			sentence = re.sub(' '+x+'\Z',' ', sentence)

		# Delete 'laughs'
		sentence = re.sub('ha(ha)+',' ', sentence)

		sentence = re.sub('\s\s+', ' ', sentence)

		return sentence

	def get_tweetpart(self, word, string, pos_of_string):
		"""" Get substring of tweet containing word split with delimiter. Return substring"""
		delimiter = '\.\s|\,|\?|!|;'
		short =re.split(delimiter,string)
		posshort = self.split_pos(short, pos_of_string)

		listindex = [item for item in range(len(short)) if 'vanavond' in short[item].lower()]

		string = ''	
		poslist = []
		test = []	
		for index in listindex:
			shortarray =short[index].split()
			string += " ".join(shortarray)
			poslist += posshort[index]
		return (string, poslist)

	def split_pos(self, splitted_array, pos_of_string):	
		""" Split POS array in same way as splitted array """
		index = 0
		splitted_pos = []
		for substring in splitted_array:
			item = substring.split()
			length = len(item)
			positem = pos_of_string[index:length+index]
			index += len(item) +1			# +1 for delimiter
			splitted_pos.append(positem)
		return splitted_pos
			


	def write_dict(self): 
		""" Write dictionary to file for analysis"""
		print "WRITING"
		c = csv.writer(open("postags_dict_saturday.csv", "wb"),delimiter = '\t')
		headings = (["POSTAG","VALUES"])
		c.writerow(headings)

		sorted_pos = sorted(self.dictionary_pos.iteritems(), key=operator.itemgetter(1))
		for item in sorted_pos:
			row = []
			row.append(item[0])
			row += set(item[1])
			c.writerow(row)

	def write_tofile(self, tweetarray):
		""" Write tweets, shortened tweets and postags to file for analysis """
		c = csv.writer(open("shortened_day2_test.csv", "wb"),delimiter = ',')
		headings = (["Tweet","Shorttweet"])
		c.writerow(headings)
		for index,item in enumerate(self.tweets):
			row = [item, tweetarray[index]]
			c.writerow(row)
			row2 = ['', self.short_pos[index]]
			c.writerow(row2)

	def load_stopword_file(self):
		stopword_file = open( "dutch-stop-words.txt", "r" )
		array = []
		for line in stopword_file:
			word = re.sub('\n','', line)
			array.append( word )
		self.STOPWORD_FILE = array

	def create_totalcorpus(self, ngramarray, array):
		""" Create corpus for all specified ngrams"""
		self.corpus = {}
		for ngram in ngramarray:
			self.corpus.update(self.create_pos_corpus(ngram,m.short_pos))
			self.corpus[('BW(vanavond)', u'VZ(init)')] = 0.0
		print sorted(self.corpus.iteritems(), key=operator.itemgetter(1), reverse=True)[:10]
		for index,item in enumerate(self.short_tweets):
			self.print_out_version(self.short_tweets[index], self.short_pos[index])
	

m = TopicExtraction(True)

m.create_wordpostuples(m.tweets)

#m.read_from_file()
m.begin_ngram_dictionary(1)
m.write_dict()
m.get_unzippedtuples_array(m.tuples)
m.tweetparts()
m.write_tofile(m.short_tweets)
m.create_totalcorpus([2,3,4], m.short_pos) 

#m.gensim_test()