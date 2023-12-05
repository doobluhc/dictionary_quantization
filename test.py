import torch 
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
img_idx = [3,500,2433,1000,4000]
coco_val = dset.CocoCaptions(root = 'mscoco/val2017',
                        annFile = 'mscoco/annotations/captions_val2017.json')
                    
for idx in img_idx:
    print(coco_val[idx][1][2])
# pic = np.array(coco_val[0][0])
# print(pic.shape)
# res = resize(pic,(512, 512, 3))
# print(res.shape)
# im = Image.fromarray((res * 255).astype(np.uint8))
# im.save('test1.png')
# print(pic.shape)




    

