import cv2
import numpy
from numpy.core.numeric import True_
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConfidence = detectionConfidence
        self.trackConfidence = trackConfidence
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode,self.maxHands,
                                        self.detectionConfidence,self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4,8,12,16,20]


    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        #for creating hand connections
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

        return img



    def findPosition(self,img,handNo=0,draw=True):
        xList = []
        yList = []
        bbox = []   #List for boundary
        self.landmark_list=[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] 
            for id, lm in enumerate(myHand.landmark):
                #print(img.shape)
                img_height, img_width, img_channel = img.shape
                pixel_x, pixel_y = int(lm.x*img_width), int(lm.y*img_height)
                xList.append(pixel_x)
                yList.append(pixel_y)
                #print(id, pixel_x, pixel_y)
                self.landmark_list.append([id,pixel_x, pixel_y])
                #print(id, lm)

                #Detecting the tip of the index finger
                #if id == 8:
                if draw:
                    cv2.circle(img, (pixel_x, pixel_y), 5, (255,0,255), cv2.FILLED)
                
            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            bbox = xMin, yMin, xMax, yMax

            if draw:
                cv2.rectangle(img, (bbox[0],bbox[1]),(bbox[2],bbox[3]),(205,150,55),3)

        return self.landmark_list, bbox


    def fingersUp(self):
        fingers=[]

        #For thumb
        if self.landmark_list[self.tipIds[0]][1] > self.landmark_list[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        #For 4 fingers
        for id in range(1,5):
            if self.landmark_list[self.tipIds[id]][2] < self.landmark_list[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers


    def findDistance(self,p1,p2,img,draw=True):

        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]

        cx,cy = (x1 + x2)//2, (y1 + y2)//2

        if draw:
            cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
            cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
            cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
            cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)
        return length,img,[x1,y1,x2,y2,cx,cy]



def main():
    cTime = 0   #Current time
    pTime = 0   #Previous time
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img,1)   #Flipping image for correct handedness
        img = detector.findHands(img,draw=False)
        landmark_list = detector.findPosition(img, draw=False)
        #if len(landmark_list) !=0:
            #print(landmark_list[4])
        #for fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255),3)

        cv2.imshow("start", img)
        if cv2.waitKey(1) & 0xFF == 27:
            print("Esc pressed, closing")
            break





if __name__ == "__main__":
    main()










