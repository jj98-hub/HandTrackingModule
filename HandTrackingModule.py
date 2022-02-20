import mediapipe as mp
import cv2
import numpy as np
import math


# Index number of each finger tips (from mediapipe documentation)
thumb = 4
index = 8
middle = 12 
ring = 16 
pinky = 20 

normalFinger = [index,middle,ring,pinky]

def fingerStatus(id,HandDataDict):
    if HandDataDict[id][1] < HandDataDict[id-3][1] and id in normalFinger:
        return True
    elif HandDataDict[id][1] > HandDataDict[id-3][1] and id in normalFinger:
        return False
    elif HandDataDict[id][0] < HandDataDict[id-2][0] and id == thumb:
        return True
    elif HandDataDict[id][0] > HandDataDict[id-2][0] and id == thumb:
        return False



class HandDetector:
    def __init__(self):
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1)
        self.mpDraw = mp.solutions.drawing_utils
        self.HandData = {}
        self.ylist = [0]*21
    
    def getAllPosition(self,frame,draw = True):
        imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        result = self.hands.process(imgRGB)
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)
                for id, lm in enumerate(handLms.landmark):
                    h,w,c = imgRGB.shape
                    self.HandData[id] = (int(lm.x*w),int(lm.y*h))
                    self.ylist[id] = int(lm.y*h)
            return self.HandData
    
    def calDis(self,x1,y1,x2,y2):
        distance = int(math.sqrt((x1-x2)**2+(y1-y2)**2))
        return distance

    def fingerUp(self,HandDataDictionary,finger):
        if finger != thumb and HandDataDictionary is not None:
            palm_X = HandDataDictionary[0][0]
            palm_Y = HandDataDictionary[0][1]
            finger_X = HandDataDictionary[finger][0]
            finger_Y = HandDataDictionary[finger][1]
            knuckle_X = HandDataDictionary[finger-2][0]
            knuckle_Y = HandDataDictionary[finger-2][1]
            tip_palm_dis = self.calDis(finger_X,finger_Y,palm_X,palm_Y)
            knuckle_palm_dis = self.calDis(palm_X,palm_Y,knuckle_X,knuckle_Y)
            if tip_palm_dis > knuckle_palm_dis :
                return True
            else:
                return False
        if finger == thumb and HandDataDictionary is not None:
            thumb_X = HandDataDictionary[thumb][0]
            thumb_Y = HandDataDictionary[thumb][1]
            knuckle_X = HandDataDictionary[index-3][0]
            knuckle_Y = HandDataDictionary[index-3][1]
            knuckle2_X = HandDataDictionary[pinky-3][0]
            palm_X = HandDataDictionary[0][0]
            palm_Y = HandDataDictionary[0][1]
            if self.calDis(thumb_X,thumb_Y,palm_X,palm_Y) < self.calDis(knuckle_X,knuckle_Y,palm_X,palm_Y) or knuckle_X<thumb_X<knuckle2_X or knuckle_X>thumb_X>knuckle2_X:
                return False
            else:
                return True
    
    def detectGesture(self,HandDataDictionary):
        if not self.fingerUp(HandDataDictionary,thumb) and self.fingerUp(HandDataDictionary,index) and not self.fingerUp(HandDataDictionary,middle) and not self.fingerUp(HandDataDictionary,ring) and not self.fingerUp(HandDataDictionary,pinky):
            return 'one'
        elif not self.fingerUp(HandDataDictionary,thumb) and self.fingerUp(HandDataDictionary,index) and  self.fingerUp(HandDataDictionary,middle) and not self.fingerUp(HandDataDictionary,ring) and not self.fingerUp(HandDataDictionary,pinky):
            return 'two'
        elif not self.fingerUp(HandDataDictionary,thumb) and self.fingerUp(HandDataDictionary,index) and  self.fingerUp(HandDataDictionary,middle) and  self.fingerUp(HandDataDictionary,ring) and not self.fingerUp(HandDataDictionary,pinky):
            return 'three'
        elif not self.fingerUp(HandDataDictionary,thumb) and self.fingerUp(HandDataDictionary,index) and  self.fingerUp(HandDataDictionary,middle) and  self.fingerUp(HandDataDictionary,ring) and  self.fingerUp(HandDataDictionary,pinky):
            return 'four'
        elif self.fingerUp(HandDataDictionary,thumb) and self.fingerUp(HandDataDictionary,index) and  self.fingerUp(HandDataDictionary,middle) and  self.fingerUp(HandDataDictionary,ring) and  self.fingerUp(HandDataDictionary,pinky):
            return 'five'
        elif  self.fingerUp(HandDataDictionary,thumb) and not self.fingerUp(HandDataDictionary,index) and  not self.fingerUp(HandDataDictionary,middle) and  not self.fingerUp(HandDataDictionary,ring) and  self.fingerUp(HandDataDictionary,pinky):
            return 'six'
        elif  self.fingerUp(HandDataDictionary,thumb) and  self.fingerUp(HandDataDictionary,index) and   self.fingerUp(HandDataDictionary,middle) and  not self.fingerUp(HandDataDictionary,ring) and  not self.fingerUp(HandDataDictionary,pinky):
            return 'three'
        elif  self.fingerUp(HandDataDictionary,thumb) and  not self.fingerUp(HandDataDictionary,index) and  not self.fingerUp(HandDataDictionary,middle) and  not self.fingerUp(HandDataDictionary,ring) and  not self.fingerUp(HandDataDictionary,pinky) and self.ylist.index(min(self.ylist)) == thumb:
            return 'thumbs up'
        elif  self.fingerUp(HandDataDictionary,thumb) and  not self.fingerUp(HandDataDictionary,index) and  not self.fingerUp(HandDataDictionary,middle) and  not self.fingerUp(HandDataDictionary,ring) and  not self.fingerUp(HandDataDictionary,pinky) and self.ylist.index(max(self.ylist)) == thumb:
            return 'thumbs down'





if __name__ == '__main__':
    detector = HandDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)     # comment or delete this if you are not use it as selfie mode
        data = detector.getAllPosition(frame)
        result = detector.detectGesture(data)
        print(result)        # the output will be within the outputlist shown in the last line of this script
        cv2.imshow("Camera", frame) 
        if cv2.waitKey(1) == 27:     # press escape key to stop the program
            break
    cap.release()
    cv2.destroyAllWindows()


#outputlist = ['one','two','three','four','five','six','thumbs up','thumbs down']