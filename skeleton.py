# !/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

import cv2

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/color', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            # listen image topic
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            cv2.imshow('Image',image)
            cv2.waitKey(1)
            # prepare rotate_cmd msg
            # DO NOT DELETE THE BELOW THREE LINES!
            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP
    
            # determine background color
            # TODO 
            # determine the color and assing +1, 0, or, -1 for frame_id
            red_int=image[20,10,2]
            blue_int=image[20,10,0]
            gre_int=image[20,10,1]
            if red_int>blue_int and red_int>gre_int:
                msg.frame_id ='-1'
                print('red')
            elif blue_int>red_int and blue_int>gre_int:
                msg.frame_id ='+1'
                print('blue')
            # msg.frame_id = '0'  # STOP
            # msg.frame_id = '-1' # CW 
            
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

