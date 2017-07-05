'''
	Author: Shixin Li
	Date: Jul 4th, 2017
'''

from FlowerTrainer import *
from FlowerClassifier import *
import json as js

## Load and transform the training data
def transform(input):
	with open(input) as f:
		flowers = f.readlines()
	flower_set = set()
	records = []
	for i in flowers:
		lines = i.strip().split(',')
		flower_set.add(lines[-1])
		records.append({'features':[float(x) for x in lines[:-1]], 'label':lines[-1]})
	transformed = {'flowers':list(flower_set), 'training-records':records}

	return js.dumps(transformed)



if __name__ == '__main__':
	path_to_input = raw_input("Please provide the path of the input file:\n")
	input_data = transform(path_to_input)
	trainer = FlowerTrainer()
	trainer.train(input_data)
	trainer.save('model.pkl')

	classifier = FlowerClassifier()
	classifier.load('model.pkl')
	path_to_test = raw_input("\nPlease enter the path of test data (json format):\n")
	with open(path_to_test) as f:
		raw_data = f.read()
	print '-----------------------------\n'
	print classifier.predict(raw_data)
