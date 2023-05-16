from PIL import Image
import numpy as np

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


img = Image.open('/gpfs3/scratch/xzhang31/VIGOR/NewYork/panorama/__7jkfc7WRIyUjPhm6AM7w,40.763530,-73.973548,.jpg')

cropped_im = img.crop((0, int(img.size[1]*0.5), img.size[0], img.size[1]))
cropped = np.array(cropped_im)

############################ Apply Polar Transform to Aerial Images in CVUSA Dataset #############################
S = 750  # Original size of the aerial image
height = 122  # Height of polar transformed aerial image
width = 671   # Width of polar transformed aerial image

i = np.arange(0, height)
j = np.arange(0, width)
jj, ii = np.meshgrid(j, i)

y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)


test = sample_bilinear(cropped, x, y)
Image.fromarray(np.uint8(test)).save('test.png')

###########################################################################################################################

i = np.arange(0, 1024)
j = np.arange(0, 1024)
jj, ii = np.meshgrid(j, i)

H = img.size[1] 
W = img.size[0]

r = np.tan(((H-ii)*np.pi)/H)
theta = 2*np.pi*jj/W
y = -r*np.sin(theta)
x = r*np.cos(theta)

test = sample_bilinear(cropped, x, y)

Image.fromarray(np.uint8(test)).save('test.png')








