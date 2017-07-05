'''
	Author: Shixin Li
	Date: Jul 4th, 2017
'''

import pickle
import json as js

class FlowerClassifier(object):

	## Load the model and flower names
	def load(self, path):
		with open(path, 'rb') as f:
			self.flowers, self.clf = pickle.load(f)

	## Return output
	def predict(self, input):
		data = js.loads(input)
		ids = [i['id'] for i in data['input-records']]
		X = [i['features'] for i in data['input-records']]
		y = self.clf.predict(X)
		y_flower = [self.flowers[int(i)] for i in y]

		output = {'output-records':[]}

		for i in xrange(len(X)):
			output['output-records'].append({'id':ids[i], 'label':y_flower[i]})

		return js.dumps(output)
