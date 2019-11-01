'''
record points information with ground truth sample
this is code Only executed at calibration stage

'''


import cv2
import pickle
import blob

# Read image
fileName = "data/BlobTest2.jpg"
im = cv2.imread(fileName, cv2.IMREAD_GRAYSCALE)
keypoints, im_with_keypoints = blob.detection(im)

# #key points
kpts = []
for kpt in keypoints:
    if type(kpt) is list or type(kpt) is tuple:
        print(kpt)
        kpts.append(kpt)
    else:
        print([kpt.pt,kpt.size])
        kpts.append([kpt.pt, kpt.size])

# Save blobs points
file = open("keypoints", "wb")
pickle.dump(kpts, file)
file.close()
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)