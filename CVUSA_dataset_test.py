from CVUSA_dataset import MyDataset
import cv2
import numpy as np

dataset = MyDataset()

sample = np.random.randint(0, len(dataset)-1)
item = dataset.__getitem__(sample)
jpg = item['jpg']
txt = item['txt']
hint = item['hint']

cv2.imwrite(filename=f'orign{sample}.png', img=jpg)
cv2.imwrite(filename=f'semantic{sample}.png', img=hint)