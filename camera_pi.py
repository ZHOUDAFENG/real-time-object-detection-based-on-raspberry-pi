#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  camera_pi.py
#
#
#
import time
import io
import threading
import picamera
import tensorflow as tf
import numpy as np
import cv2

# define a list of classes corresponding to trained MobileNet SSD
# ramdon generate a set of colors for bounding box of each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

class Camera(object):
    thread = None  # background thread that reads frames from camera
    frame = None  # current frame is stored here by background thread
    last_access = 0  # time of last client access to the camera

    def initialize(self):
        if Camera.thread is None:
            # start background frame thread
            Camera.thread = threading.Thread(target=self._thread)
            Camera.thread.start()

            # wait until frames start to be available
            while self.frame is None:
                time.sleep(0)

    def get_frame(self):
        Camera.last_access = time.time()
        self.initialize()
        return self.frame

    @classmethod
    def _thread(cls):
        with picamera.PiCamera() as camera:
            # camera setup
            camera.resolution = (320, 240)
            camera.hflip = True
            camera.vflip = True

            # let camera warm up
            camera.start_preview()
            time.sleep(2)

            stream = io.BytesIO()

            sess = tf.Session()

            for foo in camera.capture_continuous(stream, 'jpeg',
                                                 use_video_port=True):
                # store frame
                stream.seek(0)
                frame = stream.read()

                #convert binary image file to matrix form
                temp = tf.image.decode_jpeg(frame)
                frame_matrix = sess.run(temp)

                # get (height, width) info
                (h, w) = frame_matrix.shape[:2]

                # resize and normalization operation
                # construct an input blob for forward propagation
                blob = cv2.dnn.blobFromImage(cv2.resize(frame_matrix, (300, 300)), 0.007843, (300, 300), 127.5)

                # forward propagation
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
                        cv2.rectangle(frame_matrix, (startX, startY), (endX, endY),
                                      COLORS[idx], 2)
                        y = startY - 15 if startY - 15 > 15 else startY + 15
                        cv2.putText(frame_matrix, label, (startX, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

                #convert image matrix to binary form
                temp = tf.image.encode_jpeg(frame_matrix)
                cls.frame = sess.run(temp)

                # reset stream for next frame
                stream.seek(0)
                stream.truncate()

                # if there hasn't been any clients asking for frames in
                # the last 10 seconds stop the thread
                if time.time() - cls.last_access > 10:
                    break
        cls.thread = None

