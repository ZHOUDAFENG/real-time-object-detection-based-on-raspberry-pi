## case test  
### dependencies
* opencv >= 3
* tensorflow
* numpy


### implementation
In the terminal of raspberry pi/laptop, run:
```
python image_detection.py --input_image=test.jpg
```
## real Time Object Detection based on Laptop
### dependencies
* numpy
* opencv >= 3

### implementation
In the terminal of laptop, run:
```
python frame_stream_detection.py
```
## real Time Object Detection based on Raspberry Pi
### dependencies
* flask
* picamera
* tensorflow
* numpy
* opencv >= 3

### implementation
In the terminal of raspberry pi, run:
```
python appCam.py
```
Then use your laptop (make sure the same network segment) to visit *http:// + raspberry-pi's IP:8000*.  
**DETAILS**: See [https://liuhe76.github.io/2018/09/07/raspberry-pi-real-time-object-detection/](https://liuhe76.github.io/2018/09/07/raspberry-pi-real-time-object-detection/).