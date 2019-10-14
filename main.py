import cv2
import imutils
import argparse
import numpy as np 
from imutils.video import VideoStream

ap = argparse.ArgumentParser()
# ap.add_argument('-o', '--object_name', required=True, help='path to yolo config file')
ap.add_argument('-c', '--config', default='yolov3.cfg', help='path to yolo config file')
ap.add_argument('-w', '--weights', default='yolov3.weights', help='path to trained weights')
ap.add_argument('-cl', '--classes', default='yolov3.txt', help='path to file contain classes name')

args = ap.parse_args()

# return output



def get_output_layer(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    cv2.rectangle(img, (x,y), (x_plus_w, y_plus_h), (255,0,0), 2)
    cv2.putText(img, label, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

cap = VideoStream(src=0).start()

classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readline()]

net = cv2.dnn.readNet(args.weights, args.config)

while True:
    frame = cap.read()
    image = imutils.resize(frame, width=600)
    With = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop = False)
    net.setInput(blob)
    outs = net.forward(get_output_layer(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if (confidence > 0.5):
                center_x = int(detection[0]* With)
                center_y = int(detection[1]* Height)
                w = int(detection[2]*With)
                h = int(detection[3]*Height)
                x = center_x - w/2
                y = center_y - h/2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x,y,w,h])
    
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], round(x), round(y), round(x+w), round(y+h))

    cv2.imshow("object-detection", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.stop()
cv2.destroyAllWindows()


