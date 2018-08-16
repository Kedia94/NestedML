# Step 0: Load the Data
import pickle
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


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
from mynet import my_net, inference, inference10, inferencen, n

# placeholders
x = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 1))
y = tf.placeholder(dtype=tf.int32, shape=None)
keep_prob = tf.placeholder(tf.float32)


# training pipeline
lr = 0.001

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

def evaluate10(X_data, y_data):
    
    num_examples = X_data.shape[0]
    total_accuracy1 = 0
    total_accuracy2 = 0
    total_accuracy3 = 0
    total_accuracy4 = 0
    total_accuracy5 = 0
    total_accuracy6 = 0
    total_accuracy7 = 0
    total_accuracy8 = 0
    total_accuracy9 = 0
    total_accuracy10 = 0
    
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCHSIZE):
        batch_x, batch_y = X_data[offset:offset+BATCHSIZE], y_data[offset:offset+BATCHSIZE]
        accuracy1 = sess.run(accuracy_operation1, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy2 = sess.run(accuracy_operation2, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy3 = sess.run(accuracy_operation3, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy4 = sess.run(accuracy_operation4, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy5 = sess.run(accuracy_operation5, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy6 = sess.run(accuracy_operation6, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy7 = sess.run(accuracy_operation7, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy8 = sess.run(accuracy_operation8, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy9 = sess.run(accuracy_operation9, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        accuracy10 = sess.run(accuracy_operation10, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy1 += accuracy1 * len(batch_x)
        total_accuracy2 += accuracy2 * len(batch_x)
        total_accuracy3 += accuracy3 * len(batch_x)
        total_accuracy4 += accuracy4 * len(batch_x)
        total_accuracy5 += accuracy5 * len(batch_x)
        total_accuracy6 += accuracy6 * len(batch_x)
        total_accuracy7 += accuracy7 * len(batch_x)
        total_accuracy8 += accuracy8 * len(batch_x)
        total_accuracy9 += accuracy9 * len(batch_x)
        total_accuracy10 += accuracy10 * len(batch_x)
        
    return total_accuracy1 / num_examples, total_accuracy2 / num_examples, total_accuracy3 / num_examples, total_accuracy4 / num_examples, total_accuracy5 / num_examples, \
            total_accuracy6 / num_examples, total_accuracy7 / num_examples, total_accuracy8 / num_examples, total_accuracy9 / num_examples, total_accuracy10 / num_examples

def evaluaten(X_data, y_data):
    
    num_examples = X_data.shape[0]
    accuracy = []
    total_accuracy = []
    for i in range(0, n):
        accuracy.append(0)
        total_accuracy.append(0)
    
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCHSIZE):
        batch_x, batch_y = X_data[offset:offset+BATCHSIZE], y_data[offset:offset+BATCHSIZE]
        for i in range(0, n):
            accuracy[i] = sess.run(accuracy_operation[i], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            total_accuracy[i] += accuracy[i] * len(batch_x)
    for i in range(0, n):
        total_accuracy[i] /= num_examples
        
    return total_accuracy
import time
DO_TRAIN = 1	# 1 for get accuracy, 0 for get inference time profiling
DO10 = 2		# 2 for n nested, 1 for 10 nested, 0 for original

with tf.device('/gpu:0'):
    if DO10 == 2:
        logits = inferencen(x, n_classes=n_classes, keep_prob=keep_prob, is_training=True)
        cross_entropy = []
        loss_functions = []
        for i in range(0, n):
            cross_entropy.append(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[i], labels=y))
            loss_functions.append(tf.reduce_mean(cross_entropy[i]))
        loss_function = 0
        for i in range(0, n):
            loss_function += loss_functions[i]
        loss_function /= n

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.minimize(loss=loss_function)

        correct_prediction = []
        for i in range(0, n):
            correct_prediction.append(tf.equal(tf.argmax(logits[i], 1), tf.cast(y, tf.int64)))

        accuracy_operation = []
        for i in range(0, n):
            accuracy_operation.append(tf.reduce_mean(tf.cast(correct_prediction[i], tf.float32)))

    elif DO10 == 1:
        logits1, logits2, logits3, logits4, logits5, \
            logits6, logits7, logits8, logits9, logits10 = inference10(x, n_classes=n_classes, keep_prob=keep_prob, is_training=True)
        cross_entropy1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits1, labels=y)
        cross_entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits2, labels=y)
        cross_entropy3 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits3, labels=y)
        cross_entropy4 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits4, labels=y)
        cross_entropy5 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits5, labels=y)
        cross_entropy6 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits6, labels=y)
        cross_entropy7 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits7, labels=y)
        cross_entropy8 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits8, labels=y)
        cross_entropy9 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits9, labels=y)
        cross_entropy10 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits10, labels=y)
        loss_function1 = tf.reduce_mean(cross_entropy1)
        loss_function2 = tf.reduce_mean(cross_entropy2)
        loss_function3 = tf.reduce_mean(cross_entropy3)
        loss_function4 = tf.reduce_mean(cross_entropy4)
        loss_function5 = tf.reduce_mean(cross_entropy5)
        loss_function6 = tf.reduce_mean(cross_entropy6)
        loss_function7 = tf.reduce_mean(cross_entropy7)
        loss_function8 = tf.reduce_mean(cross_entropy8)
        loss_function9 = tf.reduce_mean(cross_entropy9)
        loss_function10 = tf.reduce_mean(cross_entropy10)
 
        loss_function = (loss_function1+ loss_function2+ loss_function3 + loss_function4 + loss_function5 + \
                         loss_function6 + loss_function7 + loss_function8 + loss_function9 + loss_function10)/10

        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        train_step = optimizer.minimize(loss=loss_function)


# Question 3
# metrics and functions for model evaluation
        correct_prediction1 = tf.equal(tf.argmax(logits1, 1), tf.cast(y, tf.int64))
        correct_prediction2 = tf.equal(tf.argmax(logits2, 1), tf.cast(y, tf.int64))
        correct_prediction3 = tf.equal(tf.argmax(logits3, 1), tf.cast(y, tf.int64))
        correct_prediction4 = tf.equal(tf.argmax(logits4, 1), tf.cast(y, tf.int64))
        correct_prediction5 = tf.equal(tf.argmax(logits5, 1), tf.cast(y, tf.int64))
        correct_prediction6 = tf.equal(tf.argmax(logits6, 1), tf.cast(y, tf.int64))
        correct_prediction7 = tf.equal(tf.argmax(logits7, 1), tf.cast(y, tf.int64))
        correct_prediction8 = tf.equal(tf.argmax(logits8, 1), tf.cast(y, tf.int64))
        correct_prediction9 = tf.equal(tf.argmax(logits9, 1), tf.cast(y, tf.int64))
        correct_prediction10 = tf.equal(tf.argmax(logits10, 1), tf.cast(y, tf.int64))
        accuracy_operation1 = tf.reduce_mean(tf.cast(correct_prediction1, tf.float32))
        accuracy_operation2 = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))
        accuracy_operation3 = tf.reduce_mean(tf.cast(correct_prediction3, tf.float32))
        accuracy_operation4 = tf.reduce_mean(tf.cast(correct_prediction4, tf.float32))
        accuracy_operation5 = tf.reduce_mean(tf.cast(correct_prediction5, tf.float32))
        accuracy_operation6 = tf.reduce_mean(tf.cast(correct_prediction6, tf.float32))
        accuracy_operation7 = tf.reduce_mean(tf.cast(correct_prediction7, tf.float32))
        accuracy_operation8 = tf.reduce_mean(tf.cast(correct_prediction8, tf.float32))
        accuracy_operation9 = tf.reduce_mean(tf.cast(correct_prediction9, tf.float32))
        accuracy_operation10 = tf.reduce_mean(tf.cast(correct_prediction10, tf.float32))

    else:
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


# create a checkpointer to log the weights during training
#checkpointer = tf.train.Saver()

# Qestion 3 part 2
# training hyperparameters
    BATCHSIZE = 128
    EPOCHS = 30
    BATCHES_PER_EPOCH = 5000

# Question 3 part 3
# start training# start 
    if DO_TRAIN > 0:
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            start_time = time.time()
            f = open("result.txt", 'w', 1)
            for epoch in range(EPOCHS):

                print("EPOCH {} ...".format(epoch + 1))
                f.write("EPOCH {} ...\n".format(epoch + 1))

                batch_counter = 0
                for batch_x, batch_y in image_datagen.flow(X_train_norm, y_train, batch_size=BATCHSIZE):

                    batch_counter += 1
                    sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                    if batch_counter == BATCHES_PER_EPOCH:
                        break

        # at epoch end, evaluate accuracy on both training and validation set
                if DO10 == 2:
                    train_accuracy = evaluaten(X_train_norm, y_train)
                    val_accuracy = evaluaten(X_val_norm, y_val)
                    print('Spent {:.3f}'.format(time.time()-start_time))
                    for i in range(0, n):
                        print('Train Accuracy{} = {:.3f} - Validation Accuracy{}: {:.3f}'.format(i+1, train_accuracy[i], i+1, val_accuracy[i]))
                    f.write('Spent {:.3f}\n'.format(time.time()-start_time))
                    for i in range(0, n):
                        f.write('Train Accuracy{} = {:.3f} - Validation Accuracy{}: {:.3f}\n'.format(i+1, train_accuracy[i], i+1, val_accuracy[i]))
                elif DO10 == 1:
                    train_accuracy1, train_accuracy2, train_accuracy3, train_accuracy4, train_accuracy5, \
                        train_accuracy6, train_accuracy7, train_accuracy8, train_accuracy9, train_accuracy10 = evaluate10(X_train_norm, y_train)
                    val_accuracy1, val_accuracy2, val_accuracy3, val_accuracy4, val_accuracy5, \
                        val_accuracy6, val_accuracy7, val_accuracy8, val_accuracy9, val_accuracy10 = evaluate10(X_val_norm, y_val)
                    print('Spent {:.3f}'.format(time.time()-start_time))
                    print('Train Accuracy1 = {:.3f} - Validation Accuracy1: {:.3f}'.format(train_accuracy1, val_accuracy1))
                    print('Train Accuracy2 = {:.3f} - Validation Accuracy2: {:.3f}'.format(train_accuracy2, val_accuracy2))
                    print('Train Accuracy3 = {:.3f} - Validation Accuracy3: {:.3f}'.format(train_accuracy3, val_accuracy3))
                    print('Train Accuracy4 = {:.3f} - Validation Accuracy4: {:.3f}'.format(train_accuracy4, val_accuracy4))
                    print('Train Accuracy5 = {:.3f} - Validation Accuracy5: {:.3f}'.format(train_accuracy5, val_accuracy5))
                    print('Train Accuracy6 = {:.3f} - Validation Accuracy6: {:.3f}'.format(train_accuracy6, val_accuracy6))
                    print('Train Accuracy7 = {:.3f} - Validation Accuracy7: {:.3f}'.format(train_accuracy7, val_accuracy7))
                    print('Train Accuracy8 = {:.3f} - Validation Accuracy8: {:.3f}'.format(train_accuracy8, val_accuracy8))
                    print('Train Accuracy9 = {:.3f} - Validation Accuracy9: {:.3f}'.format(train_accuracy9, val_accuracy9))
                    print('Train Accuracy10 = {:.3f} - Validation Accuracy10: {:.3f}'.format(train_accuracy10, val_accuracy10))
                    f.write('Spent {:.3f}\n'.format(time.time()-start_time))
                    f.write('Train Accuracy1 = {:.3f} - Validation Accuracy1: {:.3f}\n'.format(train_accuracy1, val_accuracy1))
                    f.write('Train Accuracy2 = {:.3f} - Validation Accuracy2: {:.3f}\n'.format(train_accuracy2, val_accuracy2))
                    f.write('Train Accuracy3 = {:.3f} - Validation Accuracy3: {:.3f}\n'.format(train_accuracy3, val_accuracy3))
                    f.write('Train Accuracy4 = {:.3f} - Validation Accuracy4: {:.3f}\n'.format(train_accuracy4, val_accuracy4))
                    f.write('Train Accuracy5 = {:.3f} - Validation Accuracy5: {:.3f}\n'.format(train_accuracy5, val_accuracy5))
                    f.write('Train Accuracy6 = {:.3f} - Validation Accuracy6: {:.3f}\n'.format(train_accuracy6, val_accuracy6))
                    f.write('Train Accuracy7 = {:.3f} - Validation Accuracy7: {:.3f}\n'.format(train_accuracy7, val_accuracy7))
                    f.write('Train Accuracy8 = {:.3f} - Validation Accuracy8: {:.3f}\n'.format(train_accuracy8, val_accuracy8))
                    f.write('Train Accuracy9 = {:.3f} - Validation Accuracy9: {:.3f}\n'.format(train_accuracy9, val_accuracy9))
                    f.write('Train Accuracy10 = {:.3f} - Validation Accuracy10: {:.3f}\n'.format(train_accuracy10, val_accuracy10))
                else:
                    train_accuracy1, train_accuracy2, train_accuracy3 = evaluate(X_train_norm, y_train)
                    val_accuracy1, val_accuracy2, val_accuracy3 = evaluate(X_val_norm, y_val)
                    print('Spent {:.3f}'.format(time.time()-start_time))
                    print('Train Accuracy1 = {:.3f} - Validation Accuracy1: {:.3f}'.format(train_accuracy1, val_accuracy1))
                    print('Train Accuracy2 = {:.3f} - Validation Accuracy2: {:.3f}'.format(train_accuracy2, val_accuracy2))
                    print('Train Accuracy3 = {:.3f} - Validation Accuracy3: {:.3f}'.format(train_accuracy3, val_accuracy3))
                    f.write('Spent {:.3f}\n'.format(time.time()-start_time))
                    f.write('Train Accuracy1 = {:.3f} - Validation Accuracy1: {:.3f}\n'.format(train_accuracy1, val_accuracy1))
                    f.write('Train Accuracy2 = {:.3f} - Validation Accuracy2: {:.3f}\n'.format(train_accuracy2, val_accuracy2))
                    f.write('Train Accuracy3 = {:.3f} - Validation Accuracy3: {:.3f}\n'.format(train_accuracy3, val_accuracy3))
            f.close()
            # log current weights
#                checkpointer.save(sess, save_path='../checkpoints/traffic_sign_model.ckpt', global_step=epoch)

    else:
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_time = time.time()
        
            for epoch in range(EPOCHS):
                batch_x, batch_y = X_val_norm[0:0+BATCHSIZE], y_val[0:0+BATCHSIZE]
                print("EPOCH {} ...".format(epoch + 1))
                if DO10 == 2:
                    for i in range(0, n):
                        a = time.time()
                        for epepep in range(0, BATCHSIZE, 1):
                            _ = sess.run(accuracy_operation[i], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                        aaa = time.time() - a
                        print('lv{}: {:.4f}, '.format(i+1, aaa/BATCHSIZE), end='')
                    print('')
                elif DO10 == 1:
                    a = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation1, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    aaa = time.time() - a
                    b = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation2, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    bbb = time.time() - b
                    c = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation3, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    ccc = time.time() - c
                    d = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation4, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    ddd = time.time() - d
                    e = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation5, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    eee = time.time() - e
                    f = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation6, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    fff = time.time() - f
                    g = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation7, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    ggg = time.time() - g
                    h = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation8, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    hhh = time.time() - h
                    i = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation9, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    iii = time.time() - i
                    j = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation10, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    jjj = time.time() - j
                    print('lv1: {:.4f}, lv2: {:.4f}, lv3: {:.4f}, lv4: {:.4f}, lv5: {:.4f}, lv6: {:.4f}, lv7: {:.4f}, lv8: {:.4f}, lv9: {:.4f}, lv10: {:.4f}'.format(aaa/BATCHSIZE, bbb/BATCHSIZE, ccc/BATCHSIZE, ddd/BATCHSIZE, eee/BATCHSIZE, fff/BATCHSIZE, ggg/BATCHSIZE, hhh/BATCHSIZE, iii/BATCHSIZE, jjj/BATCHSIZE))
                else:
                    a = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation1, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    aaa = time.time() - a
                    b = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation2, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    bbb = time.time() - b
                    c = time.time()
                    for epepep in range(0, BATCHSIZE, 1):
                        _ = sess.run(accuracy_operation3, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    ccc = time.time() - c
                    print('lv1: {:.4f}, lv2: {:.4f}, lv3: {:.4f}'.format(aaa/BATCHSIZE, bbb/BATCHSIZE, ccc/BATCHSIZE))

