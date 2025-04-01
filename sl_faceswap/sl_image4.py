import paddle
import cv2
import numpy as np
import os
import glob
import random
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
import json

class FaceswapParser:
    def __init__(self):
        """Initialize models"""
        paddle.set_device("gpu:1")
        self.faceswap_model = FaceSwap(use_gpu=True)
        self.id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
        self.id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))
        self.id_net.eval()
        self.landmarkModel = LandmarkModel(name='landmarks')
        self.landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))

    def get_id_emb(self,id_net, id_img_path):
        id_img = cv2.imread(id_img_path)
        id_img = cv2.resize(id_img, (112, 112))
        id_img = cv2paddle(id_img)
        mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
        std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
        id_img = (id_img - mean) / std

        id_emb, id_feature = id_net(id_img)
        id_emb = l2_norm(id_emb)
        return id_emb, id_feature

    def get_id_emb_from_image(self, id_net, id_img):
        id_img = cv2.resize(id_img, (112, 112))
        id_img = cv2paddle(id_img)
        mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
        std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
        id_img = (id_img - mean) / std
        id_emb, id_feature = id_net(id_img)
        id_emb = l2_norm(id_emb)
        return id_emb, id_feature
    
    def chooseRandomImage(self, type):
        if type == "filter1":
            file_path_type = ["data/source/asian_m/*.png", "data/source/asian_m/*.jpg"]
        elif type == "filter2":
            file_path_type = ["data/source/asian_g/*.png", "data/source/asian_g/*.jpg"]
        elif type == "filter3":
            file_path_type = ["data/source/western_m/*.png", "data/source/western_m/*.jpg"]
        elif type == "filter4":
            file_path_type = ["data/source/western_g/*.png", "data/source/western_g/*.jpg"]
        elif type == "filter5":
            file_path_type = ["data/source/black_m/*.png", "data/source/black_m/*.jpg"]
        elif type == "filter6":
            file_path_type = ["data/source/black_g/*.png", "data/source/black_g/*.jpg"]
        else:
            file_path_type= ["data/source/*.png", "data/source/*.jpg"]
        
        images = glob.glob(file_path_type[0]) + glob.glob(file_path_type[1])
        while True:
            source_img_random = random.choice(images)
            if not source_img_random.endswith('_aligned.png'):
                return source_img_random
    
    def parsing(self, image_data, type_value=None, need_align=True):
        """
        Process image and detect faces
        Args:
            image_data: Input image as numpy array
            type_value: Filter type
            need_align: Whether alignment is needed
        Returns:
            Tuple of (processed image, bbox list)
        """
        try:
            if need_align:
                # Process the input image directly
                target_img = image_data.copy()

                # Get bounding boxes
                bboxes = self.landmarkModel.gets_bbox(target_img)
                bbox_list = []

                for bbox in bboxes:
                    # Process bounding box
                    x_min, y_min, x_max, y_max = map(int, bbox[:4])
                    bbox_dict = {
                        'x_min': x_min,
                        'y_min': y_min,
                        'x_max': x_max,
                        'y_max': y_max,
                    }
                    bbox_list.append(bbox_dict)
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    
                    # Draw bounding box
                    cv2.rectangle(target_img, (x_min, y_min), (x_max, y_max), color, 2)

                return target_img, bbox_list
            
            return image_data, []

        except Exception as e:
            print(f"Error in parsing: {str(e)}")
            return None, []
        
    def face_align(self, image_data, merge_result=False, image_size=224):
        """Align single face in image"""
        landmark = self.landmarkModel.get(image_data)
        if landmark is not None:
            aligned_img, back_matrix = align_img(image_data, landmark, image_size)
            return aligned_img, back_matrix
        return None, None
                    
    def faces_align(self, image_data, image_size=224):
        """Align multiple faces in image"""
        aligned_imgs = []
        landmarks = self.landmarkModel.gets(image_data)
        for landmark in landmarks:
            if landmark is not None:
                aligned_img, back_matrix = align_img(image_data, landmark, image_size)
                aligned_imgs.append([aligned_img, back_matrix])
        return aligned_imgs

def process_image(image_data, type_value=None):
    """
    Process image with face detection
    Args:
        image_data: Input image (numpy array)
        type_value: Filter type
    Returns:
        Tuple of (processed image, bounding boxes)
    """
    parser = FaceswapParser()
    return parser.parsing(image_data, type_value)

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        type_value = None
        if len(sys.argv) > 2:
            type_value = sys.argv[2]
        
        image = cv2.imread(image_path)
        if image is not None:
            result_img, bbox_list = process_image(image, type_value)
            if result_img is not None:
                cv2.imwrite('output.png', result_img)
                with open('bbox.json', 'w') as f:
                    json.dump(bbox_list, f, indent=4)
                print("Processing completed successfully")
            else:
                print("Failed to process image")
        else:
            print(f"Failed to load image: {image_path}")