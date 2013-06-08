import operator

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

	tweets = []
	summaries = []
	sumdict = {}

	def __init__(self, tweets, summaries):
		"""	Initialize tweets for use"""
		self.tweets = tweets
  		self.summaries = summaries


	def create_dictionary(self):
		"""Create dictionary for summary and frequencies"""
		for item in self.summaries:
			self.sumdict[item] = self.sumdict.get(item,0) +1

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
			
		
"""
TESTING
tweets = [['test dit test dit'], ['proberen'], ['nog meer tweets testen'], ['nog meer tweets testen']]
summaries = ['testen', 'uitproberen', 'testen', 'testen', 'no', 'uitproberen', 'la', 'la']
r = Ranking(tweets, summaries)
r.create_dictionary()
test = r.create_frequencylist()
r.print_popular(4, test)
"""
