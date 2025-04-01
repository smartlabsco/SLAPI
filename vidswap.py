import os
#os.chdir('./sl_faceswap')
print(os.getcwd())

import sys
sys.path.append('/home/smartlabs/apitest/sl-parsing-api/sl_faceswap')
sys.path.append('/home/smartlabs/apitest/sl-parsing-api')
import sl_faceswap.sl_video as sli\

import sl_faceswap.sl_video as sli
import sys






def imgswap():
    imgparser = sli.FaceswapParser()
    imgparser.parsing('../UPLOAD_FILE.mp4')

if __name__ == '__main__':
    imgswap()