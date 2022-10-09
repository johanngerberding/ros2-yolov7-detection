import cv2 
import numpy as np 
import random  
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Image
from bboxes_ex_msgs.msg import BoundingBoxes, BoundingBox 


from cv_bridge import CvBridge, CvBridgeError
import onnxruntime as ort 


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    scale_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        scale_ratio = min(scale_ratio, 1.0)

    new_unpad = int(round(shape[1] * scale_ratio)), int(round(shape[0] * scale_ratio)) 
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    # divide padding into 2 sides
    dw /= 2
    dh /= 2   

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    # add border 
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, scale_ratio, (dw, dh) 


class ObjectDetector(Node):
    def __init__(self, onnx_path, cuda=False, debug=False):
        super().__init__('object_detector')
        self.bridge = CvBridge() 
        self.sub = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.detection_callback, 
            10,
        )
        self.sub
        self.pub = self.create_publisher(BoundingBoxes, 'object_detection/bboxes', 10) 
        self.provider = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=self.provider)
        self.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
            'hair drier', 'toothbrush']
        self.colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(self.names)} 
        self.debug = debug 

    def detection_callback(self, msg):
        try: 
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough') 
        except CvBridgeError as e: 
            self.get_logger().info(e) 

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255 
        
        outname = [i.name for i in self.session.get_outputs()]
        inname = [i.name for i in self.session.get_inputs()]
        inp = {inname[0]: im}
        out = self.session.run(outname, inp)[0]
        out_msg = BoundingBoxes()  
        
        for (batch_id, x0, y0, x1, y1, cls_id, score) in out:
            bbox = BoundingBox()
            box = np.array([x0, y0, x1, y1]) 
            box -= np.array(dwdh * 2)
            box /= ratio
            box = box.round().astype(np.int32).tolist() 
            bbox.xmin = max(int(box[0]), 0)
            bbox.ymin = max(int(box[1]), 0)
            bbox.xmax = max(int(box[2]), 0)
            bbox.ymax = max(int(box[3]), 0)
            bbox.class_id = self.names[int(cls_id)]
            bbox.probability = float(score)
            out_msg.bounding_boxes.append(bbox)
        print(type(out_msg)) 
        self.pub.publish(out_msg)


def main(args=None):
    print('Hi from object_detection.')
    rclpy.init(args=args) 
    detector = ObjectDetector(onnx_path="/home/johann/dev/yolov7/checkpoints/yolov7-tiny.onnx")
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
