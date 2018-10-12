import sys
import os

sys.path.append('./')

from yolo.net.yolo_net import YoloNet, n
import tensorflow as tf 
import cv2
import numpy as np

classes_name =  ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]

#model = 'models/train/yolo1/model.ckpt-385000'
#model = 'models/train/yolo2/model.ckpt-320000'
#model = 'models/train/yolo3/model.ckpt-405000'
model = 'models/train/yolo4/model.ckpt-1000000'


directory = '/home/wjjang/NestedML/tensorflow-yolo/analysis/mAP/predicted'
targetlist = '/home/wjjang/external/ObjectDetectionDataset/voc/2007_test.txt'
threshold = 0.001

def process_predicts(predicts):
  process_list = []
#  print(predicts.shape)
  p_classes = predicts[0, :, :, 0:20]
  C = predicts[0, :, :, 20:22]
  coordinate = predicts[0, :, :, 22:]

  p_classes = np.reshape(p_classes, (7, 7, 1, 20))
  C = np.reshape(C, (7, 7, 2, 1))

  P = C * p_classes

  #print P[5,1, 0, :]

  index = np.argmax(P)
  prob = index
#  print(P.shape)
#  print(P)
#  print(index)
#  print('===============')
  index = np.unravel_index(index, P.shape)

#  print(P[index[0], index[1], index[2], index[3]])
  for i in range(P.shape[0]):
    for j in range(P.shape[1]):
      for k in range(P.shape[2]):
        for l in range(P.shape[3]):
          if (P[i,j,k,l] > threshold):
#            print('Hi:' + str(P[i,j,k,l]))

            class_num = l

            coordinate = np.reshape(coordinate, (7, 7, 2, 4))

            max_coordinate = coordinate[i, j, k, :]

            xcenter = max_coordinate[0]
            ycenter = max_coordinate[1]
            w = max_coordinate[2]
            h = max_coordinate[3]

            xcenter = (index[1] + xcenter) * (448/7.0)
            ycenter = (index[0] + ycenter) * (448/7.0)

            w = w * 448
            h = h * 448

            xmin = xcenter - w/2.0
            ymin = ycenter - h/2.0

            xmax = xmin + w
            ymax = ymin + h

            process_list.append([xmin, ymin, xmax, ymax, class_num, P[i,j,k,l]])

  class_num = index[3]

  coordinate = np.reshape(coordinate, (7, 7, 2, 4))

  max_coordinate = coordinate[index[0], index[1], index[2], :]

  xcenter = max_coordinate[0]
  ycenter = max_coordinate[1]
  w = max_coordinate[2]
  h = max_coordinate[3]

  xcenter = (index[1] + xcenter) * (448/7.0)
  ycenter = (index[0] + ycenter) * (448/7.0)

  w = w * 448
  h = h * 448

  xmin = xcenter - w/2.0
  ymin = ycenter - h/2.0

  xmax = xmin + w
  ymax = ymin + h

  return xmin, ymin, xmax, ymax, class_num, prob, process_list

common_params = {'image_size': 448, 'num_classes': 20, 
                'batch_size':1}
net_params = {'cell_size': 7, 'boxes_per_cell':2, 'weight_decay': 0.0005}

net = YoloNet(common_params, net_params, test=True)

image = tf.placeholder(tf.float32, (1, 448, 448, 3))
predicts = net.inference(image)

sess = tf.Session()

saver = tf.train.Saver(net.trainable_collection)

saver.restore(sess, model)

for ii in range(0, n):
  direc = os.path.dirname(directory + '/level_'+str(ii)+'/')
#  print(direc)
  if not os.path.exists(direc):
    os.makedirs(direc)

fp = open(targetlist, 'r')
lit = fp.read().splitlines()
thisnum = 0

for img in lit:
  print(str(thisnum) + " / " + str(len(lit)))
  thisnum += 1
#  print(img)
#  ii = 1
#  output_name = directory + '/level_' + str(ii) + '/' + img.split('/')[-1].split('.')[0] + '.txt'
#  output_file = open(output_name, 'w')
#  output_file.close()

#  img = '/media/wjjang/Samsung T3/ObjectDetectionDataset/voc/VOCdevkit/VOC2007/JPEGImages/000081.jpg'
#  continue
  for ii in range(0, n):
#    np_img = cv2.imread('cat.jpg')
    np_img = cv2.imread(img)
    output_name = directory + '/level_' + str(ii) + '/' + img.split('/')[-1].split('.')[0] + '.txt'
    output_file = open(output_name, 'w')
    resized_img = cv2.resize(np_img, (448, 448))
    np_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
  
    np_img = np_img.astype(np.float32)
  
    np_img = np_img / 255.0 * 2 - 1
    np_img = np.reshape(np_img, (1, 448, 448, 3))
  
    np_predict = sess.run(predicts[ii], feed_dict={image: np_img})
  
    xmin, ymin, xmax, ymax, class_num, prob, process_list = process_predicts(np_predict)
    class_name = classes_name[class_num]
    
    for i in range(len(process_list)):
      [xmin, ymin, xmax, ymax, class_num, accuracy] = process_list[i]
      class_name = classes_name[class_num]
#      print(str(i+1)+ ": "+ class_name + " " + str(xmin) +" " + str(ymin) +" "+ str(xmax) +" "+ str(ymax) + ": "+str(accuracy))
      cv2.rectangle(resized_img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255))
      cv2.putText(resized_img, class_name, (int(xmin), int(ymin)), 2, 1.5, (0, 0, 255))
      output_file.write(class_name + " " + str(accuracy) + " " + str(xmin) +" " + str(ymin) +" "+ str(xmax) +" "+ str(ymax) + '\n')
    output_file.close()
#    print("-------------" + str(ii)+ "-------------")
    
    
# print max only
#    output_file.write(class_name + " " + str(prob) + " " + str(xmin) +" " + str(ymin) +" "+ str(xmax) +" "+ str(ymax) + '\n')
    cv2.imwrite('cat_out' + str(ii) + '.jpg', resized_img)
#  print(class_name + " " + "?" + " " + str(int(xmin)) + " " + str(int(ymin)) + " " + str(int(xmax)) + " " + str(int(ymax)))

sess.close()
