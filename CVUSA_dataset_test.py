from CVUSA_dataset import MyDataset

dataset = MyDataset()
print(len(dataset))

item = dataset.__getitem__(2)
jpg = item['jpg']
txt = item['txt']
hint = item['hint']
print(txt)
print(jpg.shape)
print(hint.shape)
