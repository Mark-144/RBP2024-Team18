# !/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

import cv2
import numpy as np

##########################################################################
##################    Modified Code    ###################################

before_image=None
most_color=np.array([0,0,0])

def auto_canny(image, sigma=0.33):
    image = cv2.GaussianBlur(image, (5, 5), 0) #(3,3)

    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

def find_rectangle(image):
    img=image.copy()
    image=cv2.GaussianBlur(image,(5,5),1)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    h,s,v=cv2.split(image)
    s=np.where(s<30,255,s)

    s= np.where(s<100 , 0, np.where(s>=100 , 255, s))
    v = np.where(v<80, 0, np.where(v>=80, 255, v))
    image_msk = cv2.bitwise_and(s,v)

    image = cv2.bitwise_and(img,img,mask=image_msk)
    cv2.imshow('g',image)
    cv2.waitKey(1)
    edges=auto_canny(image)
    return image

    corners =cv2.goodFeaturesToTrack(image_msk,maxCorners=4, qualityLevel=0.01, minDistance=200)
    if corners is not None:
        corners=np.int0(corners)
        for corner in corners:
            x,y=corner.ravel()
            cv2.circle(image,(x,y),5,(0,255,0),-1)

    

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    try:
        max_contour = max(contours, key=cv2.contourArea)
    except:
        return None

    epsilon = 0.04 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)
    if len(approx) != 4:
        inverted_binary = cv2.bitwise_not(edges)
        contours, _ = cv2.findContours(inverted_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_contour = max(contours, key=cv2.contourArea)
        rotateRect = cv2.minAreaRect(max_contour)
        box=cv2.boxPoints(rotateRect)
        box=np.int0(box)

        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)
        
        if len(approx) < 4:
            return None
        # print(rotateRect)
        # approx=box
    return [approx[0][0], approx[1][0], approx[2][0], approx[3][0]]

def detect(image):
    global most_color
    global before_image

    img=image.copy()
    if before_image is None:
        print('start')
    else:
        h1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        h2=cv2.cvtColor(before_image,cv2.COLOR_BGR2GRAY)
        hist1 = cv2.calcHist([h1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([h2], [0], None, [256], [0, 256])
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        #print(similarity)
        if similarity<0.97:
            most_color=[0,0,0]
            print('changed')
    before_image=img
    H,W,C=img.shape # 행, 열, 타입(R,G,B)

    edge=[]

    y_pos = H
    x_pos = W
    app=find_rectangle(image)
    return most_common_color(app)
    if app is None:
        return image

    edge=[app[0],app[3],app[1],app[2]]

    original_coord = np.float32(edge)
    warped_coord = np.float32([[0, 0], [x_pos, 0], [0, y_pos], [x_pos, y_pos]])
 
    mat = cv2.getPerspectiveTransform(original_coord, warped_coord)
 
    warped_img = cv2.warpPerspective(img, mat, (x_pos, y_pos))
    cv2.imshow('f',warped_img)
    cv2.waitKey(1)

    return most_common_color(warped_img)

def most_common_color(image):
    global most_color
    Csum=np.sum(most_color)
    if Csum>10:
        bP=most_color[0]/Csum
        rP=most_color[1]/Csum
        oP=most_color[2]/Csum
        if rP>0.7:
            return 'R'
        elif bP>0.7:
            return 'B'
        elif oP>0.7:
            return 'Other'
    img_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    img_mask_b = cv2.inRange(img_hsv,(120-20,30,30),(120+20,255,255))
    img_mask_r1 = cv2.inRange(img_hsv,(0,30,30),(10,255,255))
    img_mask_r2 = cv2.inRange(img_hsv,(170,30,30),(255,255,255))
    img_mask_black=cv2.inRange(img_hsv,(0,0,0),(20,20,20))
    img_mask_all=cv2.inRange(img_hsv,(0,0,0),(255,255,255))

    blue=np.count_nonzero(img_mask_b ==255)
    red=np.count_nonzero(img_mask_r1==255)+np.count_nonzero(img_mask_r2==255)
    other=np.count_nonzero(img_mask_all==255)-red-blue-np.count_nonzero(img_mask_black==255)

    color_count={'B':blue,'R':red,'Other':other}
    color=max(color_count, key=color_count.get)

    if color=='B': most_color[0]+=1
    elif color=='R': most_color[1]+=1
    else: most_color[2]+=1

    return color

##########################################################################
##########################################################################
class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            # listen image topic
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            # prepare rotate_cmd msg
            # DO NOT DELETE THE BELOW THREE LINES!
            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP
            color=detect(image)

            if color=='B':
                msg.frame_id ='+1'  #CCW
            elif color=='R':
                msg.frame_id ='-1'  # CW
            elif color=='Other':
                msg.frame_id = '0'  # STOP


            # publish color_state
            self.color_pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)


if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()
