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

"""
Ideeen: Regex over "," and replace with \t
maybe: a","b replace with a\tb 
FIND OUT: REGEX characters for this!

Get N tweets from each hour until hour is between 18 and 09? 
--> This can be annotated


Create function to tokenize tweets!
NLP over tweets? Stemmer?


"""

def loadFile2(filename):
	"""
	Load corpus and split text with parameter split.
	Arguments: filename (string)
	"""
	file = open(filename,'r')
	buffer = file.read()

	# split substituted text by given splitparameter
	# BEGINNING DOES NOT GET <#SA>
	filtered_buffer = re.sub(r'","|^"|"$|"\n','<#SA>',buffer)
	return filtered_buffer

def writetofile(data):
	with open('test.csv', 'w') as fp:
		a = csv.writer(fp, delimiter='|')
		a.writerows(data)

def createfile(testset, number):
	splitset = testset.split('<#SA>"')
	testsplitset = []
	for item in splitset:
		testitem = re.sub(r'\n', '/n', item)
		testsplitset.append(testitem)

	tweets = []
	for item in testsplitset:
		sentence = []
		test = item.split("<#SA>")
		for a in test:
			if a != '' and a != '/n':
				sentence.append(a)
		tweets.append(sentence)
	return tweets



def loadFile(filename,nr_tweets):
	"""
	Load corpus and split text with parameter split.
	Arguments: filename (string), split (string)
	"""

	datafile = open(filename,'r')
	array = []
	counter = 0
	lasttime = "09"
	for line in datafile:
		# split data of tweet
		# TODO: PROBLEM WITH CHARACTERS IN TWEET IT
		linearray = line.split('","')
		print linearray
		# get time
		[date, time] = linearray[2].split(" ")
		timehour = time.split(":")[0]
		if (timehour == lasttime and counter != nr_tweets):
			counter += 1
    		#array.append(linearray)
		else:
			pass
		if (counter == nr_tweets and timehour != lasttime):
			counter = 0
			lasttime = timehour
		else:
			pass
	return array

def printtoFile(dataset, filename):
	"""
	Prints dataset to file
	input: dataset, filename
	"""
	# check if file exists
	if( os.path.isfile(filename)):
		# prompt user for overwriting
		answer = raw_input("File exists. Overwrite? (y/n)\t")
		# do not overwrite
		if (answer.lower() != "y"):
			return

	# write to file
	ofile  = open(filename, "wb")
	writer = csv.writer(ofile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_ALL)
	for row in dataset:
		writer.writerow(row)
	print "Written to file."

def loadFile3(filename, nr_tweets, total):
	"""
	Load file to gain requested tweets from file
	input: filename, tweets required per date & time, total number of tweets
	"""
	DELIMITER = ","
	with open(filename, 'rU') as open_file:
		csv_reader = csv.reader(open_file, delimiter=DELIMITER)
		# counters
		totalcounter = 0	# total tweet counter
		count = 0			# counter for tweets per time/date
		lasttime = 9		# begin time		
		array = []			# tweetarray
		for i, row in enumerate(csv_reader):
			# total tweets is reached	
			if(totalcounter == total):
				print "Done"
				break
			else:
				# Only use rows with good start of tweet (digits)
				if (len(row) == 5 and row[0].isdigit()):	
					# Only get tweets between correct time
					time = int(row[2].split(" ")[1].split(":")[0])
					if (time >= 9 and time <= 18):
						if(lasttime == time):
							count +=1
							# only add if not total nr_tweets of time
							if(count <= nr_tweets ):
								array.append(row)
								totalcounter +=1
							# new time
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



def main(argv):
	"""
	Program entry point.
	Arguments: filename, number of tweets per time, total nr of tweets
	"""

	if (len(argv) == 5):
		# Load corpus
		testset = loadFile3(argv[1], int(argv[2]), int(argv[3]))
		printtoFile(testset, argv[4])
	elif (len(argv) == 2):
		testset = loadFile2(argv[1])
		data = createfile(testset, 10)
		writetofile(data)
	else:
		print "Error: Incorrect arguments"


if __name__ == '__main__':
	sys.exit(main(sys.argv))
