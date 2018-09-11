import numpy as np
import cv2

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

# define a Video class, use the 0 one
cap = cv2.VideoCapture(0)
while True:
    # capture a frame
    ret, frame = cap.read()

    # get (height, width) info
    (h, w) = frame.shape[:2]

    # resize and normalization operation
    # construct an input blob for forward propagation
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    #forward propagation
    net.setInput(blob)
    detections = net.forward()

    # circulation for displaying each object
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence associated with the prediction
        confidence = detections[0, 0, i, 2]

        # weak detections will be filtered if the value is lower than threshold
        # in this case, threshold is set 0.5
        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            print(label)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    # show output frame with annotation
    cv2.imshow("frame streams with object detection", frame)

    # press 'q' to quit and save the last frame as an image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("capture.jpg", frame)
        break

# release camera and destroy windows
cap.release()
cv2.destroyAllWindows()