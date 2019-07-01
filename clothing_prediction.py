from tensorflow import keras
import tensorflow as tf 
import numpy as np

# Callback model to cut gradient descent if loss function is sufficiently low
class CallbackModel(tf.keras.callbacks.Callback):
	def __init__(self, callback_threshold):
		self.callback_threshold = callback_threshold

	def on_epoch_end(self, epoch, logs={}):
		if logs.get("loss") < self.callback_threshold:
			print("Canceling training data. Loss is sufficiently low")
			self.model.stop_training = True



# Training Data
def get_fashion_data():
	fashion_mnist = keras.datasets.fashion_mnist
	(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
	
	#Normalize data
	train_images = train_images / 255.0
	test_images = test_images / 255.0
	
	return train_images, train_labels, test_images, test_labels


def build_model(train_images, train_labels):
	#Model Setup
	model = keras.Sequential()
	callbacks = CallbackModel(callback_threshold=0.30)
	flattener = keras.layers.Flatten(input_shape=(28,28)) # 28 x 28 pixel images... want to flatten to single vector
	densor1 = keras.layers.Dense(512, activation= tf.nn.relu) # Hidden layer (use ReLU activation)
	densor2 = keras.layers.Dense(10, activation=tf.nn.softmax) # 10 classes of clothing, so we want last layer to only have 10 units

	#Model Add Components
	model.add(flattener)
	model.add(densor1)
	model.add(densor2)
	model.compile(optimizer = tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy")	
	
	#Model Fit
	model.fit(train_images, train_labels, epochs=15, callbacks=[callbacks])

	#Model Save (optional)
	# model.save("clothing_predictor.h5")
	# del model

	return model


def evaluate_model(model, test_images, test_labels):
	evaluate = model.evaluate(test_images, test_labels)
	print("Model accuracy is: {}".format(evaluate))

	return evaluate


def predict_data(model, test_images):
	prediction = model.predict(test_images)
	print("Prediction for tested image(s) is/are: {}".format(prediction))

	return prediction



if __name__ == '__main__':
	train_images, train_labels, test_images, test_labels = get_fashion_data()
	model = build_model(train_images, train_labels)

	evaluate_model(model, test_images, test_labels)
	predict_data(model, test_images)

	