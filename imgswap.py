import os
#os.chdir('./sl_faceswap')
print(os.getcwd())

import sys
sys.path.append('/home/smartlabs/apitest/sl-parsing-api/sl_faceswap')
sys.path.append('/home/smartlabs/apitest/sl-parsing-api')
import sl_faceswap.sl_image as sli\

import sl_faceswap.sl_image as sli
import sys






def imgswap():
    print("inside")
    imgparser = sli.FaceswapParser()
    imgparser.parsing('../UPLOAD_IMG.png')

if __name__ == '__main__':
    imgswap()