import tensorflow as tf
import pandas as pd
import random
from collections import deque

size = 6

column_names = []
my_feature_columns = []
for i in range(size):
	key = "feature_%d" % i
	my_feature_columns.append(tf.feature_column.numeric_column(key=key))
	column_names.append(key)

def main():
	print("Let's play!")
	print("Choose a number between 1 and 3.")
	textInput = ""
	
	total = 0
	win = 0
	lose = 0
	draw = 0
	
	historic = []#deque(maxlen=2*size)
	
	while (textInput != "exit"):
		textInput = input("It's your turn: ")
		try:
			player = int(textInput)
			if (player < 1 or player > 3):
				continue
		except:
			continue
		total += 1
		
		if (total > 2*size):			
			computer = play(predict(historic))
			historic.pop(0)
			historic.pop(0)
		else:
			computer = random.randint(1, 3)
		print("My choice: %d" % computer)
		
		historic.append(player)
		historic.append(computer)
		
		result = getWinner(player, computer)
		
		if (result == 0):
			draw += 1
			print("It's a draw")
		elif (result == 1):
			win += 1
			print("You win")
		elif (result == -1):
			lose += 1
			print("You lose")
		else:
			print("error")
		
		print("Statistics: win %f%%, draw %f%%, lose %f%%, total %d" % (100*win/total, 100*draw/total, 100*lose/total, total))
	
def getWinner(player, computer):
	if (player == 3 and computer == 1):
		return -1
	if (player == 1 and computer == 3):
		return 1
	return player - computer

def generateTrainingSet(historic):
	x = []
	y = []
	for i in range(size, len(historic), 2):
		x.append(historic[i - size:i])
		y.append(historic[i] - 1)
		
	train_x = pd.DataFrame(data=x,columns=column_names)
	train_y = pd.Series(data=y,name="class")
	
	return train_x, train_y
	
def play(predict):
	if (predict == 3):
		return 1
	else:
		return predict + 1
	
	
def predict(historic):
	train_x, train_y = generateTrainingSet(historic)
	# Build 2 hidden layer DNN with 10, 10 units respectively.
	classifier = tf.estimator.DNNClassifier(
		feature_columns=my_feature_columns,
		# Two hidden layers of 10 nodes each.
		hidden_units=[10, 10],
		# The model must choose between 3 classes.
		n_classes=3)
	# Train the Model.
	classifier.train(
		input_fn=lambda:train_input_fn(train_x, train_y, 100), steps=1000)
	
	predict_x = {}
	
	for idx, val in enumerate(column_names):
		predict_x[val] = [historic[len(historic) - size + idx]]
	
	predictions = classifier.predict(input_fn=lambda:eval_input_fn(predict_x,labels=None,batch_size=100))
	
	for pred_dict, expec in zip(predictions, ['1']):
		class_id = pred_dict['class_ids'][0]
		probability = pred_dict['probabilities'][class_id]
		
		template = ('\nPrediction is "{}" ({:.1f}%)"')
		print(template.format(class_id + 1, 100 * probability))
		return class_id + 1

def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)

	# Return the dataset.
	return dataset


def eval_input_fn(features, labels, batch_size):
	"""An input function for evaluation or prediction"""
	features=dict(features)
	if labels is None:
		# No labels, use only features.
		inputs = features
	else:
		inputs = (features, labels)

	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices(inputs)

	# Batch the examples
	assert batch_size is not None, "batch_size must not be None"
	dataset = dataset.batch(batch_size)

	# Return the dataset.
	return dataset

if __name__ == '__main__':
	main()
