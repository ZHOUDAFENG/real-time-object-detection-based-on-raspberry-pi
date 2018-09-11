import numpy as np
import tensorflow as tf
import cv2

# commmand line arp
tf.app.flags.DEFINE_string('input_image','','designate an image for detection')
FLAGS = tf.app.flags.FLAGS

# define a list of classes corresponding to trained MobileNet SSD
# ramdon generate a set of colors for bounding box of each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load caffe model
print("loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
print('loading successfully!')

# load the input image and record (height,width) info
# resize and normalization operation for loaded image
# construct an input blob for forward propagation
image = cv2.imread(FLAGS.input_image)
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)

# pass the blob through the network and obtain the detections and
# predictions
print("forward propagation...")
net.setInput(blob)
detections = net.forward()

# circulation for displaying each object
for i in np.arange(0, detections.shape[2]):
    # extract the confidence associated with the prediction
    confidence = detections[0, 0, i, 2]

    # weak detections will be filtered if the value is lower than threshold
    # in this case, threshold is set 0.5
    if confidence > 0.5:
        # extract the index of the class label from the `detections`,
        # then compute the (x, y)-coordinates of the bounding box for
        # the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # display the prediction
        label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
        print(label)
        cv2.rectangle(image, (startX, startY), (endX, endY),
            COLORS[idx], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

# # show the output image
# cv2.imshow("output (press any key to exit)", image)
# cv2.waitKey(0)

#write image
cv2.imwrite('output.jpg',image)


