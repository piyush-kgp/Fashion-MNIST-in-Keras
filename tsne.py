import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.examples.tutorials import mnist
from tensorflow.contrib.tensorboard.plugins import projector

LOG_DIR = 'logs/'
MNIST_DIR = 'MNIST/'
metadata = os.path.join(LOG_DIR, 'labels.tsv')
ckpt_path = os.path.join(LOG_DIR, 'images.ckpt')

#A: MNIST
mnist = mnist.input_data.read_data_sets(MNIST_DIR)
images = tf.Variable(mnist.test.images, name = 'images')
with open(metadata, 'w') as f:
    for row in mnist.test.labels:
        f.write('%s\n' %row)

# from sklearn.decomposition import PCA
# pca = PCA(n_components=50, random_state = 123, svd_solver = 'auto')
# emb = pca.fit_transform(mnist.test.images[:1000])
# with open('logs/embeddings.tsv', 'w') as f:
#     for row in emb:
#         f.write('\t'.join(list(map(str, row.tolist()))))
#         f.write('\n')

#B: FASHION MNIST
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# images = tf.Variable(np.reshape(x_test (-1, 784)), name = 'images')
# mapping = {
# 0:	'T-shirt/top',
# 1:	'Trouser',
# 2:	'Pullover',
# 3:	'Dress',
# 4:	'Coat',
# 5:	'Sandal',
# 6:	'Shirt',
# 7:	'Sneaker',
# 8:	'Bag',
# 9:	'Ankle boot'
# }
# with open(metadata, 'w') as f:
#     for row in y_test:
#         f.write('%s\t' %mapping[row])

#Common Part
with tf.Session() as sess:
    saver = tf.train.Saver([images])
    sess.run(images.initializer)
    saver.save(sess, ckpt_path)
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    embedding.metadata_path = 'labels.tsv'
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)
