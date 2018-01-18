#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#Implementation of the k nearest neighbour algorithm from scratch
#in python using standard library.

import csv
import random
import math
import time

def load_dataset(filename, split_ratio, training_set, test_set):
	'''
	Loader function of iris data set.
	load_dataset(filename, split_ratio, training_set, test_set) -> returns None.
	'''
	with open(filename,'r') as iris_file:
		lines = csv.reader(iris_file)
		dataset = list(lines)
		
		#last columns of the five columns is the class name.
		for x in range(len(dataset)):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
			
#		random.shuffle(dataset)
#		shuffle_point = int(split_ratio * len(dataset))
#		training_set.extend(dataset[:shuffle_point])
#		test_set.extend(dataset[shuffle_point:])
		
			
			if random.random() < split_ratio:
				training_set.append(dataset[x])
			else:
				test_set.append(dataset[x])
			
def eucleadian_distance(instance1, instance2, length):
	
	distance = 0
	for x in range(length):
		distance += (instance1[x] - instance2[x]) ** 2
	
	return math.sqrt(distance)

def get_neighbours(training_set, test_sample, k):

	distances = list() 
	length = len(test_sample) - 1
	for i in range(len(training_set)):
		dist = eucleadian_distance(test_sample, training_set[i], length)
		distances.append( (training_set[i][-1], dist) )
	
	distances.sort(key = lambda x:x[1])
	
	neighbours = list()
	
	for x in range(k):
		neighbours.append(distances[x][0])
	
	return neighbours
	
def get_response(neighbours):
	
	class_votes = dict()
	for neighbour in neighbours:
		if neighbour in class_votes:
			class_votes[neighbour] += 1
		else:
			class_votes[neighbour] = 1
	
	sorted_votes = sorted(class_votes.items(),key = lambda x:x[1])
	return sorted_votes[0][0]

def get_accuracy(test_set, predictions):
	
	correct = 0
	for i in range(len(test_set)):
		if predictions[i] == test_set[i][-1]:
			correct += 1
	
	return (correct/len(predictions))*100
		
		
	


def main():
	training_set = list()
	test_set = list()
	split_ratio = 0.7
	filename = 'iris.data'
	print('Initiating loading dataset and spliting it')
	start_time = time.time()
	load_dataset(filename, split_ratio, training_set, test_set)
	print('Dataset loaded in {}.'.format(time.time()-start_time))
	
	predictions = list()
	k = 3
	
	for i in range(len(test_set)):
		neighbours = get_neighbours(training_set, test_set[i], k)
		result = get_response(neighbours)
		predictions.append(result)
		print('Actual class {} and predicted class {}'.format(test_set[i][-1],result))
		
	accuracy_score = get_accuracy(test_set,predictions)
	
	print("Accuracy of the {} nearest neighbours is {}".format(k, accuracy_score))
		
if __name__ == '__main__':
	main()	
