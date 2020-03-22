
import numpy as np
import cv2


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

class ObjectAndFaceDetection:
    def __init__(self, option):
        #self.__inputImage = cv2.imread("car.jpg")
        self.__inputImage = cv2.imread("people.jpg")
        self.__chooseWhatToDo = option

    def __ConstructBlob(self,netModel):
        if(self.__chooseWhatToDo == 1):
            blob = cv2.dnn.blobFromImage(cv2.resize(self.__inputImage, 
                                  (300, 300)), 0.007843, (300, 300), 127.5)
        elif(self.__chooseWhatToDo == 2):
            blob = cv2.dnn.blobFromImage(cv2.resize(self.__inputImage,
                                  (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        netModel.setInput(blob)

    def __PerformObjectDetection(self, netModel, height, width):
        detections = netModel.forward()
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                classIndex = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (X1, Y1, X2, Y2) = box.astype("int")

                #label = "{}: {:2.2f}%".format(CLASSES[idx], confidence*100)
                y = Y1 - 15 if Y1 -15 > 15 else Y1 +15
                if(self.__chooseWhatToDo == 1):
                    label = "%.2f"%confidence
                    if CLASSES:
                        label = "%s:%s"%(CLASSES[classIndex],label)
                    cv2.rectangle(self.__inputImage,(X1, Y1), (X2, Y2), COLORS[classIndex], 2)
                    cv2.putText(self.__inputImage, label, (X1, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[classIndex], 2)

                elif(self.__chooseWhatToDo == 2):
                    label = "%.2f"%confidence
                    cv2.rectangle(self.__inputImage, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
                    cv2.putText(self.__inputImage, label, (X1, y), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                
  
    def ShowResults(self):
        if(self.__chooseWhatToDo == 1):
            print("Performing object detection")
            netModel = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt"
                                , "MobileNetSSD_deploy.caffemodel")
        elif(self.__chooseWhatToDo == 2):
            print("Performing face detection")
            netModel = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt"
                               , "res10_300x300_ssd_iter_140000.caffemodel")
        height = self.__inputImage.shape[0]
        width = self.__inputImage.shape[1]
        self.__ConstructBlob(netModel)
        self.__PerformObjectDetection(netModel,height,width)
        cv2.imshow("Image", self.__inputImage)
        
        cv2.waitKey(0)

objectClass =ObjectAndFaceDetection(2)
objectClass.ShowResults()