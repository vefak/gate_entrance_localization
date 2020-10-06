

import cv2

cap = cv2.VideoCapture(2) # video capture source camera (Here webcam of laptop) 

i = 0

while(True):
    ret,frame = cap.read()
    cv2.imshow('img1',frame) #display the captured image
    
    if cv2.waitKey(33) & 0xFF == 32: # SPACE
   
        cv2.imwrite('./images/00{}webcam.jpg'.format(i),frame)
        i = i+ 1
        print('./images/00{}webcam.jpg'.format(i))
        cv2.putText(frame, 'Captured..', (50,50), cv2.FONT_HERSHEY_SIMPLEX ,  
                   2, (23,200,110), 2, cv2.LINE_AA) 
                
        cv2.imshow('img1',frame)
        cv2.waitKey(1000)

    elif cv2.waitKey(33) & 0xFF == 27: # ESC 
        print("Quitting...")
        
        break
cv2.destroyAllWindows()
cap.release()