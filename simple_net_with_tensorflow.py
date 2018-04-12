import tensorflow as tf


def first_test():
	# Initialize two constants
	x1 = tf.constant([1,2,3,4])
	x2 = tf.constant([5,6,7,8])

	# Multiply
	result = tf.multiply(x1, x2)

	# Initialize Session and run `result`
	with tf.Session() as sess:
		output = sess.run(result)
		print(output)


def simple_net(train_images_28x28, train_labels, test_images_28x28, test_labels):
	# Initialize placeholders 
	x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
	y = tf.placeholder(dtype = tf.int32, shape = [None])

	# Flatten the input data
	images_flat = tf.contrib.layers.flatten(x)

	# Fully connected layer 
	logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

	# Define a loss function
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))

	# Define an optimizer 
	train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

	# Convert logits to label indexes
	correct_pred = tf.argmax(logits, 1)

	# Define an accuracy metric
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	print("images_flat: ", images_flat)
	print("logits: ", logits)
	print("loss: ", loss)
	print("predicted_labels: ", correct_pred)


	tf.set_random_seed(1234)
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(500):
			_, loss_value = sess.run([train_op, loss], feed_dict={x: train_images_28x28, y: train_labels})
			if i % 10 == 0:
				print("Loss: ", loss_value)




		import matplotlib.pyplot as plt
		import random

		# Pick 10 random images
		sample_indexes = random.sample(range(len(test_images_28x28)), 10)
		sample_images = [test_images_28x28[i] for i in sample_indexes]
		sample_labels = [test_labels[i] for i in sample_indexes]

		# Run the "correct_pred" operation
		predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

		# Print the real and predicted labels
		print(sample_labels)
		print(predicted)

		# Display the predictions and the ground truth visually.
		fig = plt.figure(figsize=(10, 10))
		for i in range(len(sample_images)):
			truth = sample_labels[i]
			prediction = predicted[i]
			plt.subplot(5, 2,1+i)
			plt.axis('off')
			color='green' if truth == prediction else 'red'
			plt.text(40, 10, "Truth:        {0}\nPrediction: {1}".format(truth, prediction), fontsize=12, color=color)
			plt.imshow(sample_images[i],  cmap="gray")

		plt.show()