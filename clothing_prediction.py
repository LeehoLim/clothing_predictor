from tensorflow import keras
import tensorflow as tf 
import numpy as np


# Callback model to cut gradient descent if loss function is sufficiently low
class CallbackModel(tf.keras.callbacks.Callback):
	def __init__(self, callback_threshold=0.30, accuracy = 0.85):
		self.callback_threshold = callback_threshold
		self.accuracy = accuracy

	def on_epoch_end(self, epoch, logs={}):
		if logs.get("acc") > self.accuracy:
			print("Canceling training data. Loss is sufficiently low")
			self.model.stop_training = True


# Training Data
def get_fashion_data():
	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	
	#Normalize data
	train_images = train_images.reshape(60000,28,28,1)
	train_images = train_images / 255.0

	test_images = test_images.reshape(10000, 28,28,1)
	test_images = test_images / 255.0
	
	return train_images, train_labels, test_images, test_labels


def build_model(train_images, train_labels):
	#Model Setup
	model = keras.Sequential()
	callbacks = CallbackModel()

	#Convolution 1
	conv_1 = keras.layers.Conv2D(64, (3,3), activation="relu", input_shape=(28,28,1))
	pool_1 = keras.layers.MaxPooling2D(2,2)

	#Convolution 2
	conv_2 = keras.layers.Conv2D(64, (3,3), activation="relu")
	pool_2 = keras.layers.MaxPooling2D(2,2)

	#Softmax for multi-classification (> binary)
	flattener = keras.layers.Flatten(input_shape=(28,28)) # 28 x 28 pixel images... want to flatten to single vector
	densor1 = keras.layers.Dense(128, activation= tf.nn.relu) # Hidden layer (use ReLU activation)
	densor2 = keras.layers.Dense(10, activation=tf.nn.softmax) # 10 classes of clothing, so we want last layer to only have 10 units

	#Model Add Components
	model.add(conv_1)
	model.add(pool_1)

	model.add(conv_2)
	model.add(pool_2)

	model.add(flattener)
	model.add(densor1)
	model.add(densor2)
	model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])	
	
	#Model Fit
	model.fit(train_images, train_labels, epochs=15, callbacks=[callbacks])

	#Model Save (optional)
	# model.save("clothing_predictor.h5")
	# del model

	return model


def evaluate_model(model, test_images, test_labels):
	evaluate = model.evaluate(test_images, test_labels)

	return evaluate


def predict(model, test_images):
	prediction = model.predict(test_images, batch_size=10)

	return prediction


if __name__ == '__main__':
	train_images, train_labels, test_images, test_labels = get_fashion_data()
	model = build_model(train_images, train_labels)
	evaluation = evaluate_model(model, test_images, test_labels)
	print("Model loss is {loss} and accuracy is {acc}.".format(loss=evaluation[0], acc=evaluation[1]))

	