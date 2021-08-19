import cv2
import time
import autopy
import HandTracking as ht
import  numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480
frameR = 70
smoothening = 5 #smoothening value
prevLocX, prevLocY = 0, 0
currLocX, currLocY = 0, 0

cap = cv2.VideoCapture(0)
cTime = 0   #Current Time
pTime = 0   #Previous Time
cap.set(3,wCam)
cap.set(4,hCam)
detector = ht.handDetector(maxHands=1)

wScr, hScr = autopy.screen.size()
#print(wScr, hScr)

#For volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume = cast(interface,POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol=0
volbar = 400
volPer = 0


while True:
    #Finding Hand Landmarks
    success, img = cap.read()
    img = cv2.flip(img,1)#Flipping image for correct hand
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)
    
    #Getting the tip of index finger and middle finger
    if len(lmList)!=0:
        #*************For Ai Virtual Mouse****************#
        x1,y1 = lmList[8][1:]   #for index finger
        x2,y2 = lmList[12][1:]  #for middle finger
        #print(x1,y1,x2,y2)

        #check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)
        #cv2.rectangle(img,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)

        #Only index finger is moving:Moving mode
        if fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
            
            #Converting co-ordinates
            x3 = np.interp(x2,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y2,(frameR,hCam-frameR),(0,hScr))

            #x3 = np.interp(x2,(0,wCam),(0,wScr))
            #y3 = np.interp(y2,(0,hCam),(0,hScr))

            currLocX = prevLocX + (x3-prevLocX) / smoothening
            currLocY = prevLocY + (y3-prevLocY) / smoothening


            #move mouse
            autopy.mouse.move(currLocX,currLocY)
            cv2.circle(img,(x1,y1),15,(0,50,40),cv2.FILLED)
            prevLocX, prevLocY = currLocX, currLocY

        #Index and middle finger are up:Clicking mode
        if fingers[1] == 1 and fingers[2] == 1:

            #Finding distance between index finger and middle finger
            length,img,lineInfo = detector.findDistance(8,12,img,draw=False)
            #print(length)
            
            #Clicking mouse if distance is short
            if length < 20:
                cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)
                autopy.mouse.click()

        #***********----------------------------------***********#

        #******************For Volume Control*******************#
        area = ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))//100
        #print(area)
        if 250<area<1000:

            #Find Distance between thumb and index finger
            length, img, lineInfo = detector.findDistance(4,8,img)
            #print(length)

            #Convert Volume
            #vol = np.interp(length,[50,300],[minVol,maxVol])
            if fingers[0]==1 and fingers[1]==1 and fingers[2]==0 and fingers[3]==0:
                volBar = np.interp(length,[50,250],[400,150])
                volPer = np.interp(length,[50,250],[0,100])

                smoothness = 2
                volPer = smoothness*round(volPer/smoothness)
                #volume.SetMasterVolumeLevel(vol,None)
                if not fingers[4]:
                    volume.SetMasterVolumeLevelScalar(volPer/100,None)
                    cv2.circle(img,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv2.FILLED)



    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #Printing fps on screen
    cv2.putText(img, str(int(fps)), (10,50), cv2.FONT_HERSHEY_PLAIN, 3, (255,255,255), 5)
    cv2.imshow("Start",img)

    #For closing the video frame.
    if cv2.waitKey(1) & 0xFF == 27:
        print("Esc pressed. Closing ...")
        break