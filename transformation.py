from PIL import Image
import numpy as np
import math

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


pano = Image.open('/gpfs3/scratch/xzhang31/VIGOR/NewYork/panorama/rTW64elYRVtD5DWJ9kBgnA,40.731011,-73.995289,.jpg')
sat = Image.open('/gpfs3/scratch/xzhang31/VIGOR/NewYork/satellite/satellite_40.7309416656_-73.9953203614.png')
sat = sat.convert('RGB')

cropped_pano = pano.crop((0, int(pano.size[1]*0.5), pano.size[0], pano.size[1]))
pano_arr = np.array(pano)
sat_arr = np.array(sat)

############################ Polar Transform #############################
S = sat.size[0]  # Original size of the aerial image
#height = 122  # Height of polar transformed aerial image
#width = 671   # Width of polar transformed aerial image
#height = pano_arr.shape[0]  # Height of polar transformed aerial image
#width = pano_arr.shape[1]    # Width of polar transformed aerial image
#
#i = np.arange(0, height)
#j = np.arange(0, width)
#jj, ii = np.meshgrid(j, i)
#
#y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
#x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)
#
#test = sample_bilinear(sat_arr, x, y)
#Image.fromarray(np.uint8(test)).save('test.png')

############################ bird's eye #############################
# print(pano.size) (width, height)
height = sat.size[0]   # Height of polar transformed aerial image
width = sat.size[1]    # Width of polar transformed aerial image
c = 100 

i = np.arange(0, 2000)
j = np.arange(0, 2000)
jj, ii = np.meshgrid(j, i)

y = (width*np.arctan((height-ii)/jj))/(2*np.pi)
x = height*(1-(np.arctan(np.sqrt(1+np.power((height-ii)/jj, 2))*jj/c))/np.pi) + 50

#y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
#x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)

print(f'jj\n{jj}\n{jj.shape}\nii\n{ii}\n{ii.shape}')
print(f'x\n{x}\n{x.shape}\ny\n{y}\n{y.shape}')

test = sample_bilinear(pano_arr, x, y)
Image.fromarray(np.uint8(test)).save('test.png')
Image.fromarray(np.uint8(pano_arr)).save('test_pp.png')
Image.fromarray(np.uint8(sat_arr)).save('test_ss.png')







