from __future__ import annotations
import torch
import torchvision.transforms as transforms
import configparser
import argparse
import os
import sys
import cv2
from pathlib import Path
import numpy as np
import glob
import json

from annotate.model import LPDetectionNet

SAVE_CWD = os.getcwd()
os.chdir(os.getcwd() + "/detection")
sys.path.append(os.getcwd())

from detection.execute import do_detect, build_detect_model

os.chdir(SAVE_CWD)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

class LPSwapParser:
    def __init__(self):
        self.args = None
        self.device = None
        self.detection_network = None
        self.annotate_network = None
        self.transform = None

    def initialize(self, cfg_dir, useGPU=True):
        """Initialize networks and configurations"""
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        config = configparser.RawConfigParser()
        config.read(cfg_dir)

        basic_config = config["basic_config"]

        # Weight Files
        args.detection_weight_file = basic_config["detection_weight_file"]
        args.annotate_weight_file = basic_config["annotate_weight_file"]

        # Input Data File
        args.source = basic_config["source"]

        # GPU Number
        args.gpu_num = basic_config["gpu_num"]

        # Detection Parameters
        args.infer_imsize_same = basic_config.getboolean('infer_imsize_same')
        args.detect_save_library = basic_config.getboolean('detect_save_library')
        args.data = basic_config.get("data", 'detection/data/AD.yaml')
        args.half = basic_config.getboolean('half')
        
        imgsz = int(basic_config.get("detect_imgsz", "640"))
        args.detect_imgsz = [imgsz]
        
        args.conf_thres = float(basic_config.get("conf_thres", "0.9"))
        args.iou_thres = float(basic_config.get("iou_thres", "0.45"))
        args.max_det = int(basic_config.get("max_det", "1000"))

        # Output Directory & Result save parameters
        args.output_dir = basic_config["output_dir"]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        # Set up GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
        device = torch.device("cuda:0" if useGPU else "cpu")

        # Build Detection Network
        args.detect_weights = args.detection_weight_file
        detection_network, imgsz, stride, names, pt = build_detect_model(args, device)

        # Add network parameters to args
        args.pt = pt
        args.stride = stride
        args.imgsz = imgsz
        args.names = names

        # Load networks
        with torch.no_grad():
            detection_network.eval()

        self.args = args
        self.device = device
        self.detection_network = detection_network
        self.transform = transforms.ToTensor()

    def detect(self, img_tensor, img_mat):
        """Detect license plates in the image"""
        img_tensor = 255 * img_tensor.permute(1, 2, 0)
        detect_preds = do_detect(self.args, self.detection_network, img_tensor, self.args.imgsz, self.args.stride, auto=True)
        return detect_preds

    def file_to_torchtensor(self, imgname):
        """Convert image file to torch tensor"""
        if not os.path.exists(imgname):
            raise FileNotFoundError(f"The specified image file does not exist: {imgname}")
        img_mat = cv2.cvtColor(cv2.imread(imgname), cv2.COLOR_BGR2RGB)
        img_tensor = self.mat_to_torchtensor(img_mat)
        return (img_mat, img_tensor)

    def mat_to_torchtensor(self, img_mat):
        """Convert numpy matrix to torch tensor"""
        img_tensor = self.transform(img_mat)
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def parsing(self, target_img_path):
        """Main parsing and license plate detection function"""
        try:
            # Load and process target image
            img_mat, img_tensor = self.file_to_torchtensor(target_img_path)
            
            # Detect license plates
            bboxes = self.detect(img_tensor, img_mat)

            # Save detected bounding boxes to JSON in the desired format
            formatted_bboxes = []
            for bbox in bboxes.detach().cpu().numpy():
                x_min = int(bbox[0])
                y_min = int(bbox[1])
                x_max = int(bbox[2])
                y_max = int(bbox[3])
                formatted_bboxes.append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max
                })

            output_json_path = './data/result/bboxes.json'
            os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
            with open(output_json_path, 'w') as json_file:
                json.dump(formatted_bboxes, json_file, indent=4)
            
            print("*********************DETECTION DONE*********************")

        except Exception as e:
            print(f"Error in parsing: {str(e)}")
            raise

def main():
    """Main execution function"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("CUDA Available:", torch.cuda.is_available())
    
    # Create necessary directories
    os.makedirs('./data/target', exist_ok=True)
    os.makedirs('./data/result', exist_ok=True)
    
    try:
        # Initialize and run parser
        parser = LPSwapParser()
        parser.initialize('detect.cfg', useGPU=True)
        target_img_path = './data/target/UPLOAD_IMG.png'
        parser.parsing(target_img_path)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()