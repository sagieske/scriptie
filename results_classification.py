import operator
import numpy as np
import re
import pylab as pl
import matplotlib.pyplot as p


class Results_classification(object):

	modes = []
	gammas = []
	c = []
	ngrams = []
	bows = []
	f1 = []
	f1_0 = []
	f1_1 = []
	precision = []
	p_0 = []
	p_1 = []
	recall = []
	r_0 = []
	r_1 = []
	



	def __init__(self):
		"""	Initialize tweets for use"""
		f = open( "results7juni.txt", "r" )
		array = []
		for line in f:
			if 'MODE' in line:
				pass
			else:
				line = re.sub( '\s\s+', '|', line ).strip()
				line = re.sub( '] ', ']|', line ).strip()
				linearray = line.split('|')
			
				if len(linearray) != 15:
					print linearray
			
				self.modes.append(linearray[0])
				self.gammas.append(float(linearray[1]))
				self.c.append(int(linearray[2]))
				self.ngrams.append(linearray[3])
				self.bows.append(int(linearray[4]))
				self.f1.append(float(linearray[5]))
				self.f1_0.append(float(linearray[6]))
				self.f1_1.append(float(linearray[7]))
				self.precision.append(float(linearray[8]))
				self.p_0.append(float(linearray[9]))
				self.p_1.append(float(linearray[10]))
				self.recall.append(float(linearray[11]))
				self.r_0.append(float(linearray[12]))
				self.r_1.append(float(linearray[13]))
		
	def get_highest_f1(self, nr):
		""" Get highest f1_score """
		array = np.array(self.f1)
		sortarray = array.argsort()[::-1][:nr]
		return sortarray

	def get_highest_precision(self, nr):
		""" Get highest f1_score """
		array = np.array(self.precision)
		sortarray = array.argsort()[::-1][:nr]
		return sortarray

	def get_highest_recall(self, nr):
		""" Get highest f1_score """
		array = np.array(self.recall)
		sortarray = array.argsort()[::-1][:nr]
		return sortarray


	def get_modes(self, mode, array):
		values = np.array(array)
		array_indices = np.where(values == mode)[0]
		return array_indices

	def print_out(self, array_indices):
		""" Print in nice index"""
		beginstring =  'MODE'.ljust(15) + '|GAMMA'.ljust(9) + '|C'.ljust(8) + '|NGRAM'.ljust(10) + '|BOW'.ljust(10)+ '|F1'.ljust(10) +'|F1_0'.ljust(10)+'|F1_1'.ljust(10) + '|Precision'.ljust(10)+ '|P_0'.ljust(10) + '|P_1'.ljust(10) + '|Recall'.ljust(10) + '|R_0'.ljust(10) + '|R_1'.ljust(10)
		print beginstring
		print '-' * len(beginstring)

		for index in array_indices:
			string = ""
			metriclist = [self.f1[index], self.f1_0[index], self.f1_1[index], self.precision[index], self.p_0[index], self.p_1[index], self.recall[index], self.r_0[index], self.r_1[index]]
			for item in metriclist:
				string+= "|%.4f".ljust(8) %item

			print self.modes[index].ljust(15) + ("|"+str(self.gammas[index])).ljust(9) + ("|"+str(self.c[index])).ljust(8) + ("|"+self.ngrams[index]).ljust(10) + ("|"+str(self.bows[index])).ljust(10) + string


	def get_mode_arrays(self, modeindices, array):
		info = []
		for index in modeindices:
			info.append(array[index])
		return info
		

	def plot(self, ngrammode):
		""" Plot all """
		#ngrammode = '[1, 2, 3]'
		tokenindices = self.get_modes('token posneg', self.modes)
		tokenindices_ngram = self.get_modes(ngrammode, self.ngrams)
		tokenindices_intersection = list((set(tokenindices) & set(tokenindices_ngram)))
		token_bows = self.get_mode_arrays(tokenindices_intersection, self.bows)

		stemindices = self.get_modes('stem posneg', self.modes)
		stemindices_ngram = self.get_modes(ngrammode, self.ngrams)
		stemindices_intersection = list((set(stemindices) & set(stemindices_ngram)))
		stem_bows = self.get_mode_arrays(stemindices_intersection, self.bows)

		lemmaindices = self.get_modes('lemma posneg', self.modes)
		lemmaindices_ngram = self.get_modes(ngrammode, self.ngrams)
		lemmaindices_intersection = list((set(lemmaindices) & set(lemmaindices_ngram)))
		lemma_bows = self.get_mode_arrays(lemmaindices_intersection, self.bows)

		posindices = self.get_modes('pos posneg', self.modes)
		posindices_ngram = self.get_modes(ngrammode, self.ngrams)
		posindices_intersection = list((set(posindices) & set(posindices_ngram)))
		pos_bows = self.get_mode_arrays(posindices_intersection, self.bows)

		# Get f1 values
		tokenf1 = self.get_mode_arrays(tokenindices_intersection, self.f1)
		stemf1 = self.get_mode_arrays(stemindices_intersection, self.f1)
		lemmaf1 = self.get_mode_arrays(lemmaindices_intersection, self.f1)
		posf1 = self.get_mode_arrays(posindices_intersection, self.f1)
		
		tx, ty = self.sort_two_arrays(token_bows, tokenf1)

		sx, sy = self.sort_two_arrays(stem_bows, stemf1)

		lx, ly = self.sort_two_arrays(lemma_bows, lemmaf1)

		px, py = self.sort_two_arrays(pos_bows, posf1)


		pl.plot(tx, ty, label="token")
		pl.plot(sx, sy,label="stem")
		pl.plot(lx, ly, label="lemma")
		pl.plot(px, py, label="pos")

	def showplot(self):
		pl.ylim([0.3,0.8])
		pl.xlim([0,160])
		pl.legend()
		pl.show()



	def get_intersection(self, array1, array2):
		""" Get and return intersection of two arrays """
		return (set(array1) & set(array2))

	def sort_two_arrays(self, X, Y):
		""" Sort both arrays according to array1 """
		tuplesx, tuplesy = zip(*sorted(zip(X,Y)))
		return (list(tuplesx), list(tuplesy))

		



test = Results_classification()  

highest = test.get_highest_f1(10)
test.print_out(highest)
	
indices = test.get_modes('lemma posneg', test.modes)
test.get_mode_arrays(indices, test.f1)
test.plot('[1, 2, 3]')
test.plot('[1, 2]')
test.showplot()
test.sort_two_arrays([1,5,2,3], [88,89,90,91])
#test.test()
"""
#test.print_out([1,5,22,400])
highest = test.get_highest_f1(10)
test.print_out(highest)
print "-"
highest = test.get_highest_precision(10)
test.print_out(highest)
print "-"
highest = test.get_highest_recall(10)
test.print_out(highest)
"""

"""
TESTING
tweets = [['test dit test dit'], ['proberen'], ['nog meer tweets testen'], ['nog meer tweets testen']]
summaries = ['testen', 'uitproberen', 'testen', 'testen', 'no', 'uitproberen', 'la', 'la']
r = Ranking(tweets, summaries)
r.create_dictionary()
test = r.create_frequencylist()
r.print_popular(4, test)
"""
