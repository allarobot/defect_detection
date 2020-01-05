import tensorflow as tf
import numpy as np
import cv2

##-------------
## get image data
image = cv2.imread("lena.jpg")
h,w,c = image.shape
shape = (1,h,w,c)
image_input = np.array(image).reshape(shape)


##--------------
## process image
img = tf.placeholder(tf.float32, shape= shape)
pool1 = tf.nn.max_pool(img,(1,2,2,1),(1,2,2,1),padding='SAME')
pool2 = tf.nn.max_pool(pool1,(1,2,2,1),(1,2,2,1),padding='SAME')
upsample1 = tf.image.resize_images(pool2,(h,w))


##---------------
## get result
with tf.Session() as sess:
    img_processed = sess.run([upsample1,pool2], feed_dict={img:image_input})

##--------------
## save resulta
print(img_processed[0].shape,img_processed[1].shape)
img_out = img_processed[0].reshape((h,w,c))
pool_out = img_processed[1].reshape((h//4,w//4,c))

cv2.imwrite("img_out_f2s2.bmp",img_out)
cv2.imwrite("pool_out_f2s2.bmp",pool_out)
# new_pool_out =cv2.resize(pool_out,(w,h),interpolation=cv2.INTER_LINEAR)
# print("raw")
# print(pool_out[0:2,0:2,0])
# print("inter by cv")
# print(new_pool_out[0:8,0:8,0])
# cv2.imwrite("defect_npool_out_f2s2.bmp",new_pool_out)