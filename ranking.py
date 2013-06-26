import operator
import csv

class Ranking(object):
	"""
	Class for ranking summaries. A dictionary can be created and sorted, after which this sorted list can be printed out in a special format.
	
	Parameters:
	----------
	tweets : array of strings
		Tweets that are ranked

	summaries: array of strings
		Summaries made of tweets.

	sumdict : dictionary with summary keys (strings) and frequency values (int)
		Dictionary used for obtaining frequency of summaries.			
	"""

	activities = []
	activity_frequency = {}

	ACTIVITYFILE = "2000test_annotated_activityextraction_activities_correctness.csv"
	DELIMITER = "\t"

	def __init__(self):
		""" Initializes tweet and class sets """
		print "** Initialize.."
		self.startcolumn = 6
		data = csv.reader(open(self.ACTIVITYFILE, 'rU'), delimiter=self.DELIMITER)
		for i, row in enumerate(data):
			if i == 0:
				pass
			if (row[self.startcolumn] != ''):
				self.activities.append(row[self.startcolumn])

		ngrams = [1,2]
		for ngram in ngrams:
			dictionary_ngram = self.begin_ngram_dictionary(ngram)
			self.activity_frequency.update(dictionary_ngram)
		
		size = float(len(self.activities))		
		#self.activity_frequency.update((x, y/size) for x, y in self.activity_frequency.items())
		sorted_activities = sorted(self.activity_frequency.iteritems(), key=operator.itemgetter(1), reverse=True)[:40]

		for index, (activity, freq) in enumerate(sorted_activities):
			stringindex = "%i" %index
			stringactivity = "%s" %str(activity)
			stringfreq = "%.2f" %freq
			string = stringindex.ljust(4) + stringactivity.ljust(30) + stringfreq
			print string
			#print "%i:\t %s \t %.2f" %(index, activity, freq)

	def begin_ngram_dictionary(self, ngramsize):
		""" Update dictionary of POS tags with ngramsize
		"""
		corpus = {}
		for item in self.activities:
			corpus.update(self.add_to_dictionary(item, ngramsize,corpus))

		if (ngramsize == 1):
			corpus[('naar',)] = 0.0
		corpus.update((x, y*ngramsize*ngramsize*ngramsize) for x, y in corpus.items())
		return corpus

	def add_to_dictionary(self, activitytweet, ngramsize, corpus):
		"""Create dictionary for n-gram size activities and frequencies"""
		tokens = activitytweet.split()
		for index in range(0, len(tokens)-ngramsize+1):
			# Create tuple
			tupleItem = (tokens[index],)
			
			for i in range(index+1,index+ngramsize):
				tupleItem = tupleItem + (tokens[i],)
			
			# Add ngrams in dictionary with addition
			corpus[tupleItem] = corpus.get(tupleItem, 0) +1		
		return corpus


	def create_frequencylist(self):
		"""Sorts dictionary on frequency"""
		sorted_sumdict = sorted(self.sumdict.iteritems(), key=operator.itemgetter(1), reverse=True)
		return sorted_sumdict
		
	def print_popular(self, nr, sorted_sumdict):
		""" Prints #nr most popular activities"""
		print "Top %i popular activities" %nr
		print "Index".ljust(8) + "Summary".ljust(15) + "freq".ljust(10)

		# no index out of bound
		if nr > len(sorted_sumdict):
			nr = len(sorted_sumdict)

		
		for index in range(0,nr):
			(summary, frequency) = sorted_sumdict[index]
			summary = summary.ljust(15)
			popularity = (str(index+1)+":").ljust(8)
			freq = ("("+str(frequency)+")").ljust(10)
			
			print popularity + summary + freq
			
		
r = Ranking()
#r.create_dictionary()
#test = r.create_frequencylist()
#r.print_popular(4, test)

