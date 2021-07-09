FROM tensorflow/tensorflow:2.4.1-gpu

RUN pip install tensorflow-datasets opencv-python==4.1.2.30
RUN apt install -y libgl1-mesa-glx libsm6 libxrender1 libfontconfig1

COPY src /workspace/code/src
COPY images /workspace/code/images
COPY train.py /workspace/code/train.py
COPY camera_app.py /workspace/code/camera_app.py

WORKDIR /workspace/code/