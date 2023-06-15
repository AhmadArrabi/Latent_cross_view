from PIL import Image, ImageOps
import numpy as np
import os
from scipy.ndimage import map_coordinates
import tqdm

def sample_within_bounds(signal, x, y, bounds):

    xmin, xmax, ymin, ymax = bounds

    idxs = (xmin <= x) & (x < xmax) & (ymin <= y) & (y < ymax)
    
    sample = np.zeros((x.shape[0], x.shape[1], signal.shape[-1]))
    sample[idxs, :] = signal[x[idxs], y[idxs], :]

    return sample


def sample_bilinear(signal, rx, ry):

    signal_dim_x = signal.shape[0]
    signal_dim_y = signal.shape[1]

    # obtain four sample coordinates
    ix0 = rx.astype(int)
    iy0 = ry.astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    
    bounds = (0, signal_dim_x, 0, signal_dim_y)

    # sample signal at each four positions
    signal_00 = sample_within_bounds(signal, ix0, iy0, bounds)
    signal_10 = sample_within_bounds(signal, ix1, iy0, bounds)
    signal_01 = sample_within_bounds(signal, ix0, iy1, bounds)
    signal_11 = sample_within_bounds(signal, ix1, iy1, bounds)

    na = np.newaxis
    # linear interpolation in x-direction
    fx1 = (ix1-rx)[...,na] * signal_00 + (rx-ix0)[...,na] * signal_10
    fx2 = (ix1-rx)[...,na] * signal_01 + (rx-ix0)[...,na] * signal_11

    # linear interpolation in y-direction
    return (iy1 - ry)[...,na] * fx1 + (ry - iy0)[...,na] * fx2


def polar_linear(img, o=None, r=None, output=None, order=1, cont=0):
    if r is None: r = img.shape[0]
    
    if output is None:
        output = np.zeros((r*2, r*2))
    elif isinstance(output, tuple):
        output = np.zeros(output)
    if o is None: o = np.array(output.shape)/2 - 0.5
    
    out_h, out_w = output.shape
    ys, xs = np.mgrid[:out_h, :out_w] - o[:,None,None]
    rs = ((ys**2+xs**2)**0.5)
    
    ts = np.arccos(xs/rs)
    ts[ys<0] = np.pi*2 - ts[ys<0]
    ts *= (img.shape[1]-1)/(np.pi*2)
    
    for c in range(img.shape[-1]):
        output = np.expand_dims(map_coordinates(img[:,:,c], (rs, ts), order=order), axis=-1)
        if c==0: output2 = output
        else: output2 = np.uint8(np.concatenate((output2,output), axis=-1))
    output2 = np.rot90(output2, k=3)
    return output2


def log_polar(signal):
    signal = signal.crop((0, int(signal.size[1]*0.38), signal.size[0], signal.size[1]))
    signal = ImageOps.flip(signal)

    height = signal.height  # Height of polar transformed aerial image
    width = signal.width    # Width of polar transformed aerial image
    
    i = np.arange(0, width)
    j = np.arange(0, height)
    ii, jj = np.meshgrid(i, j)

    # smaller c_scale means more round street
    c_scale = 1
    y = ii
    x = np.log1p(jj*c_scale)
    x = (jj.max()-jj.min())*(x-x.min())/(x.max()-x.min()) + jj.min() 
    
    signal = np.array(signal)
    signal = sample_bilinear(signal, x, y)
    img = Image.fromarray(np.uint8(polar_linear(signal, output=(1024, 1024))))

    return img

root = '/gpfs3/scratch/xzhang31/VIGOR'
target =  '/gpfs3/scratch/aarrabi/VIGOR'
city = ['Chicago', 'NewYork', 'SanFrancisco', 'Seattle']

for c in city: 
    if not os.path.exists(f'{target}/{c}/panorama'):
        os.makedirs(f'{target}/{c}/panorama')
        print(f'{target}/{c}/panorama created!')

c = city[3]
img_list = os.listdir(f'{root}/{c}/panorama')
for img_name in tqdm.tqdm(img_list):
    if img_name[-3:] == 'jpg':
        im = log_polar(Image.open(f'{root}/{c}/panorama/{img_name}'))
        im.save(f'{target}/{c}/panorama/{img_name}')



