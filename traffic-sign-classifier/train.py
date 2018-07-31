# Step 0: Load the Data
import pickle

def load_traffic_sign_data(training_file, testing_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    return train, test

# Load pickled data
train, test = load_traffic_sign_data('traffic_signs_data/train.p', 'traffic_signs_data/test.p')
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']


# Step 1: Dataset Summary & Exploration
import numpy as np

# Number of examples
n_train, n_test = X_train.shape[0], X_test.shape[0]

# What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# How many classes?
n_classes = np.unique(y_train).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples  =", n_test)
print("Image data shape  =", image_shape)
print("Number of classes =", n_classes)

# Step 2: Design and Test a Model Architecture
import cv2 

def preprocess_features(X, equalize_hist=True):

    # convert from RGB to YUV
    X = np.array([np.expand_dims(cv2.cvtColor(rgb_img, cv2.COLOR_RGB2YUV)[:, :, 0], 2) for rgb_img in X])

    # adjust image contrast
    if equalize_hist:
        X = np.array([np.expand_dims(cv2.equalizeHist(np.uint8(img)), 2) for img in X])

    X = np.float32(X)

    # standardize features
    X -= np.mean(X, axis=0)
    X /= (np.std(X, axis=0) + np.finfo('float32').eps)

    return X

X_train_norm = preprocess_features(X_train)
X_test_norm = preprocess_features(X_test)

# Question 1
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# split into train and validation
VAL_RATIO = 0.2
X_train_norm, X_val_norm, y_train, y_val = train_test_split(X_train_norm, y_train, test_size=VAL_RATIO, random_state=0)


# create the generator to perform online data augmentation
image_datagen = ImageDataGenerator(rotation_range=15.,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1)

# take a random image from the training set

# plot the original image

# plot some randomly augmented images

#Question 2
import tensorflow as tf
from mynet import my_net, inference

# placeholders
x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
y = tf.placeholder(dtype=tf.int32, shape=None)
keep_prob = tf.placeholder(tf.float32)


# training pipeline
lr = 0.001
logits = my_net(x, n_classes=n_classes, keep_prob=keep_prob)

logits1, logits2, logits3 = inference(x, n_classes=n_classes, keep_prob=keep_prob, is_training=True)
cross_entropy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=y)
cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=y)
cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits3, labels=y)
loss_function1 = tf.reduce_mean(cross_entropy1)
loss_function2 = tf.reduce_mean(cross_entropy2)
loss_function3 = tf.reduce_mean(cross_entropy3)

loss_function = (loss_function1+ loss_function2+ loss_function3)/3

optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_step = optimizer.minimize(loss=loss_function)


# Question 3
# metrics and functions for model evaluation
correct_prediction1 = tf.equal(tf.argmax(logits1, 1), tf.cast(y, tf.int64))
correct_prediction2 = tf.equal(tf.argmax(logits2, 1), tf.cast(y, tf.int64))
correct_prediction3 = tf.equal(tf.argmax(logits3, 1), tf.cast(y, tf.int64))
accuracy_operation1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
accuracy_operation2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
accuracy_operation3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))

def evaluate(X_data, y_data):
    
    num_examples = X_data.shape[0]
    total_accuracy1 = 0
    total_accuracy2 = 0
    total_accuracy3 = 0
    
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCHSIZE):
        batch_x, batch_y = X_data[offset:offset+BATCHSIZE], y_data[offset:offset+BATCHSIZE]
        accuracy1 = sess.run(accuracy_operation1, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy2 = sess.run(accuracy_operation2, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy3 = sess.run(accuracy_operation3, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy1 += accuracy1 * len(batch_x)
        total_accuracy2 += accuracy2 * len(batch_x)
        total_accuracy3 += accuracy3 * len(batch_x)
        
    return total_accuracy1 / num_examples, total_accuracy2 / num_examples, total_accuracy3 / num_examples

# create a checkpointer to log the weights during training
checkpointer = tf.train.Saver()

# Qestion 3 part 2
# training hyperparameters
BATCHSIZE = 128
EPOCHS = 30
BATCHES_PER_EPOCH = 5000

# Question 3 part 3
import time
# start training# start 
with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    start_time = time.time()

    for epoch in range(EPOCHS):

        print("EPOCH {} ...".format(epoch + 1))

        batch_counter = 0
        for batch_x, batch_y in image_datagen.flow(X_train_norm, y_train, batch_size=BATCHSIZE):

            batch_counter += 1
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

            if batch_counter == BATCHES_PER_EPOCH:
                break

        # at epoch end, evaluate accuracy on both training and validation set
        train_accuracy1, train_accuracy2, train_accuracy3 = evaluate(X_train_norm, y_train)
        val_accuracy1, val_accuracy2, val_accuracy3 = evaluate(X_val_norm, y_val)
        print('Spent {:.3f}'.format(time.time()-start_time))
        print('Train Accuracy1 = {:.3f} - Validation Accuracy1: {:.3f}'.format(train_accuracy1, val_accuracy1))
        print('Train Accuracy2 = {:.3f} - Validation Accuracy2: {:.3f}'.format(train_accuracy2, val_accuracy2))
        print('Train Accuracy3 = {:.3f} - Validation Accuracy3: {:.3f}'.format(train_accuracy3, val_accuracy3))
        
        # log current weights
        checkpointer.save(sess, save_path='../checkpoints/traffic_sign_model.ckpt', global_step=epoch)
