import sys
import time
import math
import re
import os.path
import operator
from operator import itemgetter
import itertools
from collections import Counter
import csv


class ReadTweets(object):
	""" Class blablablabla """

	# Input file reader
	DELIMITER = "\t"
	data = csv.reader(open("vanavond_day10000.csv", 'rU'), delimiter=DELIMITER)
	OUTPUTFILE = "day_output.csv"

	def __init__(self):
		print "Init"

	def main(self, totaldays):
		array = self.loadFileDays(totaldays)
		self.writetofile(array)
		


	def loadFileDays(self, totaldays):
		""" Load file to gain tweets from # days. Return array of tweets from days
		"""
		days = 0			# counter for tweets per time/date
		lasttime = 9		# begin time		
		array = []			# tweetarray
		for i, row in enumerate(self.data):
			# Only use rows with good start of tweet (digits)
			if (row[5] == '' and row[0].isdigit()):	
				print 'test'
				time = int(row[2].split(" ")[1].split(":")[0])
				if (days < totaldays ):
					if (time >= 9 and time < 18):
						array.append(row)
					else:
						days += 1
				else:
					break
			else:			# Incorrect tweetline				
				pass
		return array

	def load_tweets_nr(self,filename, nr_tweets, total):
		""" Load file to gain requested tweets from file
		"""
		# counters
		totalcounter = 0	# total tweet counter
		count = 0			# counter for tweets per time/date
		lasttime = 9		# begin time
		
		array = []			
		for i, row in enumerate(self.data):
			# total tweets is reached	
			if(totalcounter == total):
				print "Done"
				break
			else:
				# Only use rows with good start of tweet (digits)
				if (row[5] == '' and row[0].isdigit()):	
					# Only get tweets between correct time
					time = int(row[2].split(" ")[1].split(":")[0])
					if (time >= 9 and time < 18):
						if(lasttime == time):
							count +=1
							# only add if not total nr_tweets of time
							if(count <= nr_tweets ):
								array.append(row)
								totalcounter +=1

						else:
							# reset 
							lasttime = time
							count = 1
							array.append(row)
							totalcounter +=1
				# Incorrect tweetline				
				else:
					pass
		return array

	def writetofile(self,dataset):
		""" Write dataset to file. Input: dataset, filename
		"""
		# check if file exists
		if( os.path.isfile(self.OUTPUTFILE)):
			answer = raw_input("File exists. Overwrite? (y/n)\t")
			# do not overwrite
			if (answer.lower() != "y"):
				return

		# write to file
		ofile  = open(self.OUTPUTFILE, "wb")
		writer = csv.writer(ofile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
		for row in dataset:
			writer.writerow(row)
		print "Written to file."

r = ReadTweets()
r.main(1)
