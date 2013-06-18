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

class TopicExtraction(object):
	# Read file
	DELIMITER = "\t"
	data = csv.reader(open("2000test_annotated_v2.csv", 'rU'), delimiter=DELIMITER)
	class_dict = {"Y": 0, "N": 1, "U": 0}

	short_tweets = []
	tweets = []
	posdeleted_tweettokens = []
	posdeleted_tweets = []
	posdeleted_pos = []
	tuples = []
	pos_short_tweets = []
	pos_short_pos = []
	posdeleted_postypes = []
	dictionary_pos = {}

	PORTNUMBER = 1161

	EMOTICONS = ['[=:]-*[()DdPpSsOo(\|)(\$)]']
	NUMBERS = ['\d+[(\.):]\d+', '\d+']

	def __init__(self):
		""" Initializes tweet and class sets """
		print "** Initialize.."
		for i, row in enumerate(self.data):
			# Ignores header
			if(i == 0):
				pass
			else:
			# TEMP, for testing only
				begin = 1
				end = 2000
				if (i >= begin and i <= end):
					# TEMP, Only add if class is known! for testing only
					if (self.class_dict.get(row[5].upper()) is 0):
						substituted = re.sub(r'\.\.+\s', r' ', row[3])
						substituted = re.sub(r'\sen\svanavond', r'. vanavond', substituted)
						substituted = self.remove_stopwords(substituted)

						self.tweets.append(substituted)

		self.dictionary_pos = defaultdict(list)


	def tryout_tuples(self, array):
		self.startFrogServer('start')			
		time.sleep(15)							# Time for startup server
		frogclient = FrogClient('localhost',self.PORTNUMBER)

		print "** START frog analysis."
		print "** Creating POS tags.. (This may take a while)"
		for item in array:
			wordpos_array = self.frog_tweets(frogclient, item)
			self.tuples.append(wordpos_array)	

		helpers.dump_to_file('te-tuples.txt', self.tuples)
		self.startFrogServer('stop')			


	def startFrogServer(self, mode):
		""" Starts/stops Frog server in seperate terminal """
		if(mode == 'start'):
			print "** Start Frog Server"
			os.system("frog -S " + str(self.PORTNUMBER) + " > /dev/null 2>&1 &")
		if(mode == 'stop'):
			print "** Close Frog Server"
			proc = subprocess.Popen(["pgrep", 'frog'], stdout=subprocess.PIPE) 
			for pid in proc.stdout: 
				os.kill(int(pid), signal.SIGTERM)
	
	def read_from_file(self):
		"""	Load array from file """
		fT = file('te-tuples.txt', "r")
		self.tuples = pickle.load(fT)


	def testout(self, ngramsize):
		for item in self.tuples:
			self.test(item, ngramsize)

	def test(self, tuple_array, ngramsize):

		for index in range(0, len(tuple_array)-ngramsize+1):
			# Create tuples
			tupleItem_words = (tuple_array[index][0],)
			tupleItem_pos = (tuple_array[index][1],)
			for i in range(index+1,index+ngramsize):
				tupleItem_words = tupleItem_words + (tuple_array[i][0],)
				tupleItem_pos = tupleItem_pos + (tuple_array[i][1],)

			if tupleItem_pos in self.dictionary_pos:
				value = self.dictionary_pos[tupleItem_pos]
				value.append(tupleItem_words)
				newvalue = list(set(value))
				self.dictionary_pos[tupleItem_pos] = newvalue
			else:
				self.dictionary_pos[tupleItem_pos] = [tupleItem_words]

	def write_dict(self): 
		c = csv.writer(open("postags_dict.csv", "wb"),delimiter = ',')
		headings = (["POSTAG","VALUES"])
		c.writerow(headings)

		sorted_pos = dict(sorted(self.dictionary_pos.iteritems(), key=operator.itemgetter(1)))
		for key in sorted_pos:
			row = []
			row.append(key)
			row += set(sorted_pos[key])
			c.writerow(row)

	def write_tofile(self, tweetarray):
		c = csv.writer(open("shortened.csv", "wb"),delimiter = ',')
		headings = (["Tweet","Shorttweet"])
		c.writerow(headings)

		for index,item in enumerate(self.tweets):
			row = [item, tweetarray[index]]
			c.writerow(row)
			row2 = ['', self.pos_short_pos[index]]
			c.writerow(row2)



			

	def frog_tweets(self, frogclient, tweet):
		"""	Use frog for processing according to mode. Return array of processed words """
		frogtweet = frogclient.process(tweet.lower())
		tuples_tweet = []
		for test in frogtweet:
			# Frog sometimes contains tuple of None
			if (None in test):
				pass
			else:
				word, lemma, morph, pos = test
				deleted = self.deletepos(word,pos)
				if (deleted):
					pass
				else:
					tuple_pos = (word,pos)
					tuples_tweet.append(tuple_pos)
		return tuples_tweet

	def get_tuple_words(self):
		for array in self.tuples:
			tupletweet = []
			tuplepos = []
			for (word, pos) in array:
				tupletweet.append(word)
				tuplepos.append(pos)
			self.posdeleted_tweettokens.append(tupletweet)
			self.posdeleted_pos.append(tuplepos)
		

		for index, item in enumerate(self.posdeleted_tweettokens):
			string = ''
			for word in item:
				string += word.encode('utf-8') + ' '
			self.posdeleted_tweets.append(string)

		for index, item in enumerate(self.posdeleted_pos):
			string = ''
			for pos in item:
				postype_array = str(pos).split('(')
				string += postype_array[0] + '-'
			self.posdeleted_postypes.append(string)
			#print string


	def deletepos(self, word, pos):
		""" Tests if POS tag belongs to list to be removed. Returns Boolean"""
		# POS tags to be removed	
		poslist_delete = ['TW(rang', 'TSW', 'BW', 'VNW', 'WW(vd', 'LID', 'VG', 'ADJ(nom', 'ADJ(dial)', 'comp', 'SPEC', 'WW(od', 'sup', 'ADJ(postnom']	
		poslist_notdelete = ['VNW(onbep,pron']		
		#poslist = []
		for item in poslist_delete:
			if item in str(pos):
				if 'vanavond' in word:		# do not delete vanavond (BW())
					pass
				else:
					for item in poslist_notdelete:
						if item in str(pos):
							return False
					return True
		return False
	
	def remove_stopwords(self,sentence):
		""" Removes stop words in sentences. Returns substituted sentence """
		"""
		wordlist = ['[Ii]k', '[Jj]ij', '[Hh]ij', '[Ww]ij', '[Zz]ij', '[Jj]ullie','[Mm]ij', '[Mm]e', '[Jj]e', '[Zz]e', '[Ww]e', '[Ww]eer', '[Ll]ekker', '[Mm]aar', '[Ee]ens', '[Dd]e', '[Hh]et', '[Ee]en', '[Dd]an', '[Dd]aarom', '[Ww]aarom', '[Dd]us', '[Dd]at', '[Ff]f', '[Ww]at', '[eE]even', '[Dd]enk', '[Ee]ven', '[=:]-*[()DdPpSsOo(\|)(\$)]', '\d+[(\.):]\d+', '\d+', '[Ll]ekker']
		"""
		wordlist = self.NUMBERS + self.EMOTICONS
		for x in wordlist:
			sentence = re.sub(' '+x+' ',' ', sentence)
			sentence = re.sub('\A'+x+' ',' ', sentence)
			sentence = re.sub(' '+x+'\Z',' ', sentence)
		return sentence

	def get_tweetpart(self, word, string, pos_of_string):
		"""" Get substring of tweet containing word split with delimiter. Return substring"""
		delimiter = '\.\s|\,|\?|!|;'
		short =re.split(delimiter,string)
		posshort = self.split_pos(short, pos_of_string)

		listindex = [item for item in range(len(short)) if 'vanavond' in short[item].lower()]
		string = ''	
		poslist = []	
		for index in listindex:
			string += short[index]
			poslist.append(posshort[index])
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
			
	def tweetparts(self):
		""" Get substrings of tweet containing vanavond """
		for index, item in enumerate(self.posdeleted_tweets):
			(stringpart, pospart) = self.get_tweetpart('vanavond', item, self.posdeleted_pos[index])
			self.pos_short_tweets.append(stringpart)
			self.pos_short_pos.append(pospart)

	def gensim_test(self):
		tokenarray = []		
		for item in self.pos_short_tweets:
			tokenitem = item.split()
			tokenarray.append(tokenitem)

		dictionary = gensim.corpora.Dictionary(tokenarray)

		dictionary.save('/tmp/testing_gensim.dict')


		corpus_mm = [dictionary.doc2bow(tokens_tweet) for tokens_tweet in tokenarray]
		gensim.corpora.MmCorpus.serialize('/tmp/testing_gensim.mm', corpus_mm) # store to disk, for later use

		lda = gensim.models.ldamodel.LdaModel(corpus=corpus_mm, id2word=dictionary, num_topics=20, update_every=1, chunksize=10000, passes=3)
		for i in range(0, 19):
			string = "topic #%i " + str(lda.print_topic(i))
			print string % i

		print lda.show_topics()
		"""
		corpus_tfidf = gensim.tfidf[corpus]
		for doc in corpus_tfidf:
			print doc
		"""

		#model = hdpmodel.HdpModel(bow_corpus, id2word=dictionary)
	

m = TopicExtraction()

#m.tryout_tuples(m.tweets)
m.read_from_file()
m.testout(1)
m.write_dict()
m.get_tuple_words()
m.tweetparts()
m.write_tofile(m.pos_short_tweets)
m.gensim_test()
