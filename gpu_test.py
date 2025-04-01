import os
import numpy as np
import logging
import logging.config
import json
import base64
import io
from faceModule.faceParser import FaceParser
from licenseModule.licenseParser import LicenseParser
from PIL import Image
from flask import Flask, request, jsonify, send_file

# log setting
log_config = json.load(open('log.json'))
logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)

# GPU SETTING
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Set the GPU 1 to use

print("Hello")

