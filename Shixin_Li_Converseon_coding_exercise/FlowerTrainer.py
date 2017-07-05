'''
	Author: Shixin Li
	Date: Jul 4th, 2017
'''

import json as js
import pickle
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class FlowerTrainer(object):

	def train(self, input, algo=None):
		self.flower = js.loads(input)
		
		flower_dic = {}
		for i, j in enumerate(self.flower['flowers']):
			flower_dic[j] = i 

		X = np.array([np.array(i['features']) for i in self.flower['training-records']])
		y = [flower_dic[i['label']] for i in self.flower['training-records']]

		## Leave one out cross validation
		## Find out the best model based on cv and then train the model
		if algo == None:
			algo = [svm.SVC(decision_function_shape='ovo'), svm.LinearSVC(), RandomForestClassifier(), KNeighborsClassifier(), LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')]
			model_name = ["One vs. One SVM with rbf kernel",
						  "One vs. rest linear SVM",
						  "Random forest",
						  "KNN",
						  "Multi-class logistic regression"]
			best_score = 0
			k = 0
			## Leave one out
			for clf in algo:
				score = []
				for i in xrange(len(X)):
					clf.fit(np.concatenate([X[:i], X[i+1:]]), np.concatenate([y[:i], y[i+1:]]))
					score.append(np.mean(clf.predict(X[i:i+1]) == y[i:i+1]))
				if np.mean(score) > best_score:
					best_score = np.mean(score)
					self.clf = clf
					best_name = model_name[k]
				k += 1
			print "\nBased on the result of cross-validation, we select %s model" % (best_name)

			## Train the model
			self.clf.fit(X,y)

		## Run the logistic regression if algo is not none
		else:
			self.clf = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg')
			self.clf.fit(X,y)

	## Save the model 
	def save(self, path):
		with open(path, 'wb') as f:
			pickle.dump([self.flower['flowers'], self.clf], f)

