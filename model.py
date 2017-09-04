import os
import yaml
import numpy as np
import tensorflow as tf


def glove_embeddings(glove_dir):
	"""
	this function returns glove word vectors in a dict of the form {'word' : <vector>}
	parameters : glove_dir is the directory location of the glove file for word embeddings/vectos
	"""
	vocab = []
	embeddings = []
	f = open(os.path.join(glove_dir, 'glove.6B.50d.txt'))
	for line in f:
		values = line.strip().split(' ')
		vocab.append(values[0])
		embeddings.append(values[1:])
	print 'loaded glove'
	f.close()
	return vocab, embeddings

if __name__ == '__main__':
	#load config file and parameters in it
	with open('config.yml') as config_file:
		config = yaml.load(config_file)
	glove_dir = config['glove_dir']
	data_file = config['data_file']
	sequence_length = config['sequence_length']
	num_classes = config['num_classes']
	config_file.close()

	#session variable
	sess = tf.Session()

	#get glove vectors
	vocab, embeddings = glove_embeddings(glove_dir)
	
	vocab_size = len(vocab)
	embedding_dim = len(embeddings[0])
	emb = np.asarray(embeddings)

	#W is the tensorflow variable that would hold the word vectors according to the id's which we can search 
	#using embedding_lookup function
	W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=True, name="W")
	embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
	embedding_init = W.assign(embedding_placeholder)
	sess.run(embedding_init, feed_dict={embedding_placeholder: emb})


	#get a sentence vector for each sentence
	x_test = 'of the to'
	arr = []
	for word in x_test.split():
		arr.append(vocab.index(word))
	input_sentence_vec = sess.run(tf.nn.embedding_lookup(W, arr))

	#open data file
	f = open(os.path.join(os.getcwd(), data_file))

	#inputs
	input_x = tf.placeholder(tf.float32, [None, sequence_length], name='input_x')
	input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
	dropout = tf.placeholder(tf.float32, name='dropout')

