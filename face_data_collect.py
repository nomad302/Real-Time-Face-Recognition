# Write a python script that captures images from webcam video stream
# Extract all images from the image frame (using haarcascades)
#Stores the face information into numpy arrays

# 1. Read and show video stream, captures images
# 2. Detect faces and show bounding box
# 3. Flatten the largest face image (gray scale)and save in the numpy array
# 4. Repeat the above for multiple people to generate data

import cv2
import numpy as np

#Init Camera
cap = cv2.VideoCapture(0)

# face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

skip = 0
face_data =[]
dataset_path = './data/'
file_name = input("Enter the name of the person : - ")

while True:
	ret, frame =cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	if ret == False:
		continue
	
	
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	faces = sorted(faces,key=lambda f:f[2]*f[3]) 

	# put bounding box
	# Pick the last face (because it is the largest face acc to area(f[2]*f[3]))
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		# Extract (Crop out the region of face) : Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))
		skip += 1
		if skip%10 == 0:
			face_data.append(face_section)
			print(len(face_data))


	cv2.imshow('Frame',frame)
	cv2.imshow('Face Section',face_section)



	# wait for user input to stop when press - q
	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert our facelist array into numpy array
face_data = np.asarray(face_data)
# Number of rows should be same as Number of faces = face_date.shape[0],column = -1 as we dont know
face_data = face_data.reshape((face_data.shape[0],-1)) 
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+".npy",face_data)
print("Data Successfully Saved at "+dataset_path+file_name+".npy")

cap.release()
cv2.destroyAllWindows()