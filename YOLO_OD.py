import cv2
import numpy as np 

class YoloDetection :
    def __init__(self) :
        self.classes_file = "YOLO_data/coco.names"
        self.net = cv2.dnn.readNet("YOLO_data/yolov3.weights", "YOLO_data/yolov3.cfg")
        self.layer_names = self.net.getUnconnectedOutLayersNames()

        with open(self.classes_file, 'r') as f:
            self.classes = f.read().strip().split('\n')

    def TestImg(self,Img,Intent) :
        ht , wd , _ = Img.shape

        blob = cv2.dnn.blobFromImage(Img, 0.00392, (416 , 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)

        outs = self.net.forward(self.layer_names)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.8 :
                    center_x, center_y, w, h = detection[:4] * np.array([wd, ht, wd, ht])
                    x, y, w, h = int(center_x - w / 2), int(center_y - h / 2), int(w), int(h)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        # print(f'Indices : {indices} , Boxes {boxes}')
        if len(indices) > 0 :
            for i in [indices]:
                # print()
                i = i[0]
                box = boxes[i]
                x, y, w, h = box
                label = str(class_ids[i])
                confidence = confidences[i]
                color = (0, 255, 0)
                import cv2 as cv
                found_intent = self.classes[int(label)].lower()
                if found_intent == Intent.lower() :
                    Img = np.array(Img)
                    cv.rectangle(Img, (x, y), (x + w, y + h), color, 2)
                    cv.putText(Img, f"{found_intent}: {confidence*100:.2f}", (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    return True,Img,[x,y,x+w,y+h]

        return False,Img,None

if __name__ == "__main__" :
    img = cv2.imread('spawn_images/img_4.jpg')
    yd = YoloDetection()
    print(yd.classes)
    # _ ,img = yd.TestImg(img,'sofa')
    # if _ == True :
    #     from matplotlib import pyplot as plt
    #     plt.imshow(img)
    #     plt.show()


"""
    Classes : 
    ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 
    'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 
    'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
    'teddy bear', 'hair drier', 'toothbrush']

    Required Classes :
    ['bed' , 'refrigerator' , 'chair' , 'tvmonitor' , 'sofa' , '']
"""