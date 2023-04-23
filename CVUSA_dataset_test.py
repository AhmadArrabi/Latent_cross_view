from CVUSA_dataset import MyDataset
import cv2
import numpy as np

dataset = MyDataset()
print(len(dataset))

sample = np.random.randint(0, len(dataset)-1)
item = dataset.__getitem__(sample)
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)

cv2.imwrite(filename=f'testinggg{sample}.png', img=hint)