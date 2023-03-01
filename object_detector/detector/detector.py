import cv2
import sys

image_path = sys.argv[1][sys.argv[1].rfind('/')+1:]
img_name = sys.argv[1].split('.')[0]
img_ext = sys.argv[1].split('.')[1]

img = cv2.imread(f'object_detector/images/{img_name}.{img_ext}')
classNames = []
classFile = 'object_detector/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'object_detector/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'object_detector/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(img.shape[0], img.shape[1])
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

classIds, confs, bbox = net.detect(img, confThreshold=0.58)

for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    print(classNames[classId-1], box[0] + box[2] // 2, box[1] + box[3] // 2)
    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
    cv2.putText(img, classNames[classId-1].upper(), (box[0]-10, box[1]-10),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, str(round(confidence*100, 2)), (box[0]+70, box[1]-10),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

cv2.imwrite(f'object_detector/images/{img_name}_output.{img_ext}', img)