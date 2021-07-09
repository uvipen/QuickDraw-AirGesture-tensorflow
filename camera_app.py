"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
import cv2
import argparse
from collections import deque
from src.model import CLASS_IDS
from src.utils import get_overlay, get_images

HAND_GESTURES = ["Open", "Closed"]
WHITE_RGB = (255, 255, 255)
GREEN_RGB = (0, 255, 0)
PURPLE_RGB = (255, 0, 127)


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Google's Quick Draw Project (https://quickdraw.withgoogle.com/#)""")
    parser.add_argument("-a", "--area", type=int, default=3000, help="Minimum area of captured object")
    parser.add_argument("-l", "--load_path", type=str, default="data/trained_models")
    parser.add_argument("-s", "--save_video", type=str, default="data/output.mp4")
    args = parser.parse_args()
    return args


def load_graph(path):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    InteractiveSession(config=config)
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(path, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    return detection_graph, sess


def detect_hands(image, graph, sess):
    input_image = graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = graph.get_tensor_by_name('detection_scores:0')
    detection_classes = graph.get_tensor_by_name('detection_classes:0')
    image = image[None, :, :, :]
    boxes, scores, classes = sess.run([detection_boxes, detection_scores, detection_classes],
                                      feed_dict={input_image: image})
    return np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes)


def predict(boxes, scores, classes, threshold, width, height, num_hands=2):
    count = 0
    results = {}
    for box, score, class_ in zip(boxes[:num_hands], scores[:num_hands], classes[:num_hands]):
        if score > threshold:
            y_min = int(box[0] * height)
            x_min = int(box[1] * width)
            y_max = int(box[2] * height)
            x_max = int(box[3] * width)
            category = HAND_GESTURES[int(class_) - 1]
            results[count] = [x_min, x_max, y_min, y_max, category]
            count += 1
    return results


def main(opt):
    graph, sess = load_graph("data/pretrained_model.pb")
    model = tf.keras.models.load_model(opt.load_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    out = cv2.VideoWriter(opt.save_video, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (640, 480))
    points = deque(maxlen=1024)
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    is_drawing = False
    is_shown = False
    predicted_class = None
    class_images = get_images("images", CLASS_IDS.values())

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        elif key == ord(" "):
            is_drawing = not is_drawing
            if is_drawing:
                if is_shown:
                    points = deque(maxlen=1024)
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                is_shown = False
        if not is_drawing and not is_shown:
            if len(points):
                canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                # Blur image
                median = cv2.medianBlur(canvas_gs, 9)
                gaussian = cv2.GaussianBlur(median, (5, 5), 0)
                # Otsu's thresholding
                _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contour_gs, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                if len(contour_gs):
                    contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]
                    # Check if the largest contour satisfy the condition of minimum area
                    if cv2.contourArea(contour) > opt.area:
                        x, y, w, h = cv2.boundingRect(contour)
                        image = canvas_gs[y:y + h, x:x + w]
                        image = cv2.resize(image, (28, 28))
                        image = np.array(image, dtype=np.float32)[None, :, :, None] / 255
                        image = tf.convert_to_tensor(image)
                        predictions = model.predict(image)
                        score = tf.nn.softmax(predictions[0])
                        predicted_class = np.argmax(score)
                        is_shown = True
                    else:
                        print("The object drawn is too small. Please draw a bigger one!")
                        points = deque(maxlen=1024)
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        # Read frame from camera
        ret, frame = cap.read()
        if frame is None:
            continue
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = predict(boxes, scores, classes, 0.6, 640, 480)

        # Check to see if any contours are found
        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            if is_drawing:
                if category == "Closed":
                    points.appendleft((x, y, 1))
                else:
                    points.appendleft((x, y, 0))
                for i in range(1, len(points)):
                    if points[i - 1] is None or points[i - 1][2] == 0 or points[i] is None:
                        continue
                    cv2.line(canvas, points[i - 1][:2], points[i][:2], WHITE_RGB, 5)
                    cv2.line(frame, points[i - 1][:2], points[i][:2], GREEN_RGB, 2)

        if is_shown:
            cv2.putText(frame, 'You are drawing', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, PURPLE_RGB, 5,
                        cv2.LINE_AA)
            frame[5:65, 490:550] = get_overlay(frame[5:65, 490:550], class_images[predicted_class], (60, 60))

        cv2.imshow("Camera", frame)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = get_args()
    main(opt)
