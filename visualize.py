import cv2
import os
import json
import random

j = random.randint(0,200)


with open("./Chicago_AerialMajorSplit/pano_label_balanced__corrected.json") as f:
    satellite_to_ground = json.load(f)

o = 0
for sat,v in satellite_to_ground.items():
    if o != j:
        o += 1
        continue
    
    ground_list = []
    delta_list = []
    sat = cv2.imread(os.path.join("/gpfs3/scratch/xzhang31/VIGOR/Chicago", "satellite", sat))
    
    index = 0
    for g, d in v.items():
        index += 1
        ground_list.append(g)
        delta_list.append(d)
        center = (int(320.0 - float(d[1])), int(320.0 + float(d[0])))
        g_processeed = g.replace('.jpg', '.png')
        sat = cv2.circle(sat, center, 10, (255, 133, 233), -1)
        grd = cv2.imread(os.path.join("/gpfs3/scratch/xzhang31/VIGOR/Chicago", "panorama", g))
        grd_processed = cv2.imread(os.path.join("/gpfs3/scratch/xzhang31/VIGOR/Chicago", "panorama", g_processeed))
        cv2.imwrite(f"{index}_grd.jpg", grd)
        cv2.imwrite(f"{index}_grd_processed.jpg", grd_processed)
        
    cv2.imwrite("satellite.jpg", sat)
    break
