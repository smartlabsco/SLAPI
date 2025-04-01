import torch
import os
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from faceModule.biseNet import BiSeNet

class FaceParser:
    def parsing(self, image, overlay=False, view=False): #이걸로 추론
        # load model
        n_classes = 19
        net = BiSeNet(n_classes=n_classes)
        net.cuda()
        net.load_state_dict(torch.load('model/79999_iter.pth'))
        net.eval()

        # parsing
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        with torch.no_grad():
            resize_img = image.resize((512, 512), Image.Resampling.BILINEAR)
            img = to_tensor(resize_img)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            pred = out.squeeze(0).cpu().numpy().argmax(0)
            # create parsed image and return it
            return self.create_image(resize_img, pred, overlay=overlay, view=view)

    def create_image(self, origin, pred, overlay=False, view=False):
        im = np.array(origin)
        vis_im = im.copy().astype(np.uint8)
        mask = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
        vis_parsing_anno = pred.copy().astype(np.uint8)
        vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
        vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

        num_of_class = np.max(vis_parsing_anno)

        for pi in range(1, num_of_class + 1):
            index = np.where(vis_parsing_anno == pi)
            if pi in (1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13):
                if overlay:
                    vis_parsing_anno_color[index[0], index[1], :] = (0, 255, 0)
                else:
                    mask[index[0], index[1], :] = (255, 255, 255)

        if overlay:
            # addWeighted parsed color and original image
            vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
            output = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
        else:
            # only show face area(other pixel black)
            output = cv2.bitwise_and(im, mask)

        # view
        if view:
            plt.figure()
            plt.imshow(output)
            plt.axis('off')
            plt.show()

        return output


