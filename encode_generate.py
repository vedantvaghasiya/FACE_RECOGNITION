import face_recognition
import cv2
import pickle
import os


# Importing student img
folderPath = 'images'
pathlist = os.listdir(folderPath)
print(pathlist) # we get all images name 

imgList = []
studentid = []
 
for path in pathlist:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    # studentid.append(os.path.splitext(path)[0])
    studentid.append(os.path.splitext(path)[0])
    # print(path)
    # print(os.path.splitext(path)[0])

    # print(len(imgList)) # we get the number of images in the folder
    
print(studentid)


def findEncodings(imgList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList

print("encoding is started")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown,studentid]
# print(encodeListKnown) 
print("encoding is completed")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved") 


