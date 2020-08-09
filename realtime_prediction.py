import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
model=tf.keras.models.load_model(r'C:\Users\abdul\mymodelmilk.h5')
cap = cv2.VideoCapture(r"C:\\Users\\abdul\\Downloads\\WhatsApp Video 2020-07-22 at 9.06.48 PM.mp4")
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
font = cv2.FONT_HERSHEY_SIMPLEX

textImage = np.zeros((300,512,3), np.uint8)
def outputWindow(className,textImage):
    textImage = np.zeros((300,512,3), np.uint8)
    cv2.putText(textImage,"Prediction : " + className, 
                    (30, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (255, 255, 255),
                    2)
    cv2.putText(textImage," made by- Abdullah Shaikh", 
                    (30,180), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1,
                    (255, 255, 255),
                    2)
    cv2.imshow('output',textImage)
    
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
    className = ""
    ret, frame = cap.read()
    if ret == True:
    
    # Display the resulting frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
       
        img = cv2.resize(frame,(150,150)) 
        cv2.imwrite('frame.png',img)
        img = cv2.imread('frame.png',1)
        
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images, batch_size=10)
        img = cv2.resize(frame,(640,354))
        
        print(classes[0])
        if classes[0]<0.5:
            #print( "is safe")
            className="SAFE"
            meme=cv2.imread(r"C:\Users\abdul\Downloads\herapherimeme.jpg")
            
        elif classes[0]>0.5:
            #print(" is a spill")
            className=" SPILL"
            meme= cv2.imread(r"C:\Users\abdul\Downloads\Raju-and-Shyam-running-after-Munnabhai.jpg")
        
        outputWindow(className,textImage)
        meme=cv2.resize(meme,(640,400))
        cv2.imshow("why??",meme)
        
        
        cv2.imshow('Frame',img)
  # Break the loop
    else: 
        break
    
# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

    