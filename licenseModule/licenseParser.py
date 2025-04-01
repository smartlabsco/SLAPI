import torch
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
import numpy as np

class LicenseParser:
    def parsing(self, image, threshold=0.5, overlay=False, view=False):
        # load model
        model = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, aux_loss=False)
        model.classifier = models.segmentation.segmentation.DeepLabHead(2048, 1)
        checkpoint = torch.load('model/model_v2.pth', map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        _ = model.eval()

        if torch.cuda.is_available():
            model.to('cuda')

        # parsing
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')

        with torch.no_grad():
            pred = model(input_batch)['out'][0]
            # create parsed image and return it
            return self.create_image(image, pred, threshold=threshold, overlay=overlay, view=view)

    def create_image(self, origin, pred, threshold=0.5, overlay=False, view=False):
        im = np.array(origin)
        vis_im = im.copy().astype(np.uint8)
        mask = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
        vis_parsing_anno = pred.cpu().numpy()[0] > threshold
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        index = np.where(vis_parsing_anno != 0)
        if overlay:
            vis_parsing_anno_color[index[0], index[1], :] = (0, 255, 0)
            # addWeighted parsed color and original image
            vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
            output = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        else:
            mask[index[0], index[1], :] = (255, 255, 255)
            # only show face area(other pixel black)
            output = cv2.bitwise_and(im, mask)

        # view
        if view:
            plt.figure()
            plt.imshow(output)
            plt.axis('off')
            plt.show()

        return output

