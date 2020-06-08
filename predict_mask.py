from keras.models import load_model
import cv2
import numpy as np

model = load_model('model-013.model')
face_clsfr=cv2.CascadeClassifier('D:/git/opencv/data/haarcascades_cuda/haarcascade_frontalface_alt.xml')

source=cv2.VideoCapture(0)
#img=cv2.imread('D:/ANN VIT/dataset/with_mask/image7.jpeg')

labels_dict={0:'no_mask',1:'with_mask'}
color_dict={0:(0,0,255),1:(0,255,0)}

while(True):

    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray, 1.05, 5)  
    # resized=cv2.resize(gray,(100,100))
    # normalized=resized/255.0
    # reshaped=np.reshape(normalized,(1,100,100,1))
    # result=model.predict(reshaped)
    
    # label=np.argmax(result,axis=1)[0]

    for x,y,w,h in faces:
    
        face_img=gray[y-10:y+w+10,x-10:x+w+10]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)
        
        label=np.argmax(result,axis=1)[0]
      
        resshow=str(labels_dict[label]+" "+str(result[0][label]))
        
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y),(x+w,y),color_dict[label],-1)
        cv2.putText(
           img, resshow, 
           (x, y-10),
           cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    # cv2.putText(
    #       img, labels_dict[label], 
    #       (90, 90),
    #       cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()