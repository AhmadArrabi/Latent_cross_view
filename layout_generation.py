# importing required modules
import requests
import json
import os
from tqdm import tqdm

root = '/gpfs2/scratch/xzhang31/VIGOR'
label_root = 'splits__corrected'
c = 'Seattle'
json_file = 'same_area_balanced_train__corrected.json'

with open(os.path.join(root, label_root, f'{c}_AerialMajorSplit', json_file), 'r') as j:
    city_dict = json.load(j)

my_loc = list(city_dict.keys())[:1500]
my_loc = [i[:-4].split('/')[2].split('_')[1:3] for i in my_loc]

# Enter your api key here
api_key = "77a582a921bf45118d1e78053fd69928"
 
# url variable store url
h, w = 640, 640
style = 'osm-carto'
zoom = 19


for sample in tqdm(my_loc):
    lat=sample[0]
    long=sample[1]

    url = f"https://maps.geoapify.com/v1/staticmap?style={style}&width={w}&height={h}&center=lonlat:{long},{lat}&format=png&zoom={zoom}&apiKey={api_key}"
    path = f'./Seattle/satellite/satellite_{lat}_{long}.png'
    r = requests.get(url)
    
    with open(path, 'wb') as f:
        f.write(r.content)
