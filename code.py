import cv2
import time


timestr = time.strftime("%Y%m%d-%H%M%S")

classfile = "coco.names"
with open(classfile, "r") as f:
    classnames = f.read().splitlines()
    print(classnames)

configpath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt" # we will write the target file even if it in the same file ex "/home/pi/desktop/od/file name(ssd..)
weightpath = "frozen_inference_graph.pb" # we will write the target file even if it in the same file ex "/home/pi/desktop/od/file name(froz..)

net = cv2.dnn_DetectionModel(weightpath, configpath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# def getobjects (img,draw = True):   to make the green box and its things to show up or not#
def getobjects(frame, thres, nms, draw = True, objects=[]):
    classIds, confs, bbox = net.detect(frame, confThreshold=thres, nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0:
        objects = classnames

    objectinfo = []
    if len(classIds) != 0:
        for classId, confidance, box in zip(classIds.flatten(), confs.flatten(), bbox):
            classname = classnames[classId - 1]
            if classname in objects:
                objectinfo.append([box, classname])

                if (draw):
                    cv2.rectangle(frame, box, color=(255, 0, 0), thickness=2)
                    cv2.putText(frame, classnames[classId-1].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1,
                        (0, 0, 255), 1)
                    cv2.putText(frame,str(round(confidance*100,2)),(box[0]+200,box[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)

    return frame, objectinfo


if __name__ == "__main__" :
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(timestr+".avi", fourcc, 15.0, (640, 480))
    capture_duration = 18

    start_time = time.time()

    while (int(time.time() - start_time) < capture_duration):

         succes, frame = cap.read()
         result,objectinfo  = getobjects(frame, .45, .2, objects=["car"]) # to remove box (img, False) to show all (img) to detect specific object or objects (img,objects["object name","object name","..."]
         print(objectinfo)
         # result = getobjects(img, False) to make the green box and its things to show up or not #
         cv2.imshow("OD-module-with-record", frame)
         out.write(frame)
         if cv2.waitKey(1) and 0xFF == ord('q'):
             break

