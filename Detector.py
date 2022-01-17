import cv2
import time
from PIL import Image 

def main_app(name):
        start_time = time.time()
        face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(f"./data/classifiers/{name}_classifier.xml")
        cap = cv2.VideoCapture(0)
        pred = 0
        while True:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:


                roi_gray = gray[y:y+h,x:x+w]

                id,confidence = recognizer.predict(roi_gray)
                confidence = 100 - int(confidence)
                if confidence > 50:
                            pred = pred + 1
                            text = name.upper()
                            font = cv2.FONT_HERSHEY_PLAIN
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)

                else:   
                            pred = pred -1
                            text = "UnknownFace"
                            font = cv2.FONT_HERSHEY_PLAIN
                            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)

            curr_time = int(time.time())
            time_elapsed = curr_time - int(start_time)
            cv2.imshow("image", frame)

            if cv2.waitKey(20) & 0xFF == ord('q') or time_elapsed == 20:
                if pred > 0 : 
                    dim =(124,124)
                    img = cv2.imread(f".\\data\\{name}\\{1}{name}.jpg", cv2.IMREAD_UNCHANGED)
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(f".\\data\\{name}\\valid{name}.jpg", resized)
                    Image1 = Image.open(f".\\2.png") 
                      
                    # make a copy the image so that the
                    # original image does not get affected
                    Image1copy = Image1.copy() 
                    Image2 = Image.open(f".\\data\\{name}\\valid{name}.jpg")
                    Image2copy = Image2.copy() 
                      
                    # paste image giving dimensions 
                    Image1copy.paste(Image2copy, (195, 114)) 
                      
                    # save the image  
                    Image1copy.save("end.png") 
                    frame = cv2.imread("end.png", 1)

                    cv2.imshow("Result",frame)
                    cv2.waitKey(5000)
                else:
                    print("Invalid User")
                break



        cap.release()
        cv2.destroyAllWindows()
        
