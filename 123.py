import tensorflow as tf
import pandas as pd
import random

size = 6

features = []
my_feature_columns = []
for i in range(size):
	key = "feature_%d" % i
	my_feature_columns.append(tf.feature_column.numeric_column(key=key))
	features.append(key)

def main():
	print("Let's play!")
	textInput = ""
	
	total = 0
	win = 0
	lose = 0
	draw = 0
	
	historic = []
	
	while (textInput != "exit"):
		textInput = input("It's your turn: ")
		try:
			player = int(textInput)
			if (player < 1 or player > 3):
				continue
		except:
			continue
		total += 1
		
		historic.append(player)
		
		if (total > 4*size):
			computer = play(predict(historic))
		else:
			computer = random.randint(1,3)
		print("My choice: %d" % computer)
		
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
	
	for i in range(size, len(historic) - size, 2):
		x.append(historic[i - size:i])
		y.append(historic[i])
		
	train_x = pd.DataFrame(data=x,columns=features)
	train_y = pd.Series(data=y,name="class")
	
	print(historic)
	print(train_x)
	print(train_y)
	
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
		input_fn=lambda:train_input_fn(train_x, train_y, 5))

def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	print(dict(features))
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
	print(dataset)
	exit()
	# Shuffle, repeat, and batch the examples.
	#dataset = dataset.shuffle(1000).repeat().batch(batch_size)

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
