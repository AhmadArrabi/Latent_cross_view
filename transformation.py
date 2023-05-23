from PIL import Image, ImageOps
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

cropped_pano = pano.crop((0, int(pano.size[1]*0.3), pano.size[0], pano.size[1]))
flipped_pano = ImageOps.flip(cropped_pano)
flip = False

if flip:
    pano_arr = np.array(flipped_pano)
else:
    pano_arr = np.array(pano)
sat_arr = np.array(sat)
######################################################################################################
height = cropped_pano.height  # Height of polar transformed aerial image
width = cropped_pano.width    # Width of polar transformed aerial image

i = np.arange(0, width)
j = np.arange(0, height)
ii, jj = np.meshgrid(i, j)

y = ii
# smaller c_scale means more round street
c_scale = 0.1 
x = np.log1p(jj*c_scale)
x = (jj.max()-jj.min())*(x-x.min())/(x.max()-x.min()) + jj.min() 

test = sample_bilinear(pano_arr, x, y)
Image.fromarray(np.uint8(test)).save('test_stretch.png')
#############################################scipy polar###############################################333
from scipy.ndimage import map_coordinates

def linear_polar(img, o=None, r=None, output=None, order=1, cont=0):
    if o is None: o = np.array(img.shape[:2])/2 - 0.5
    if r is None: r = (np.array(img.shape[:2])**2).sum()**0.5/2
    if output is None:
        shp = int(round(r)), int(round(r*2*np.pi))
        output = np.zeros(shp, dtype=img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=img.dtype)
    out_h, out_w = output.shape
    out_img = np.zeros((out_h, out_w), dtype=img.dtype)
    rs = np.linspace(0, r, out_h)
    ts = np.linspace(0, np.pi*2, out_w)
    xs = rs[:,None] * np.cos(ts) + o[1]
    ys = rs[:,None] * np.sin(ts) + o[0]
    map_coordinates(img, (ys, xs), order=order, output=output)
    return output

def polar_linear(img, o=None, r=None, output=None, order=1, cont=0):
    if r is None: r = img.shape[0]
    if output is None:
        output = np.zeros((r*2, r*2), dtype=img.dtype)
    elif isinstance(output, tuple):
        output = np.zeros(output, dtype=img.dtype)
    if o is None: o = np.array(output.shape)/2 - 0.5
    out_h, out_w = output.shape
    ys, xs = np.mgrid[:out_h, :out_w] - o[:,None,None]
    rs = (ys**2+xs**2)**0.5
    ts = np.arccos(xs/rs)
    ts[ys<0] = np.pi*2 - ts[ys<0]
    ts *= (img.shape[1]-1)/(np.pi*2)
    map_coordinates(img, (rs, ts), order=order, output=output)
    return output

#out = linear_polar(pano_arr[:,:,0])
img = np.expand_dims(polar_linear(test[:,:,0], output=(2000,2000)), axis=-1)
img2 = np.expand_dims(polar_linear(test[:,:,1], output=(2000,2000)), axis=-1)
img3 = np.expand_dims(polar_linear(test[:,:,2], output=(2000,2000)), axis=-1)
img = np.concatenate((img,img2,img3), axis=-1)
#img = polar_linear(np.array(ImageOps.flip(Image.fromarray(np.uint8(test))))[:,:,1], output=(2500,2500))


#Image.fromarray(np.uint8(out)).save('test_scipy_.png')
#Image.fromarray(np.uint8(img)).save('test_scipy_polar.png')

############################ Polar Transform #############################
#S = sat.size[0]  # Original size of the aerial image
#height = 122  # Height of polar transformed aerial image
#width = 671   # Width of polar transformed aerial image
#height = sat_arr.shape[0]  # Height of polar transformed aerial image
#width = sat_arr.shape[1]    # Width of polar transformed aerial image
#
#i = np.arange(0, height)
#j = np.arange(0, width)
#jj, ii = np.meshgrid(j, i)
#
#y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
#x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)
#
#print(x.shape, y.shape)
#test = sample_bilinear(sat_arr, x, y)
#Image.fromarray(np.uint8(test)).save('test.png')

############################ bird's eye #############################
#xp = np.arange(0, pano.width)
#yp = np.arange(0, pano.height)
#c = 100
#
#ii, jj = np.meshgrid(xp, yp) #horizontal xp, vertical yp
##ii=ii+0.001
#
#center_x, center_y = pano.width // 2, pano.height // 2
#circle_radius = 500
#circle_diameter = circle_radius * 2
#
#r = np.sqrt(ii - circle_radius ** 2 + jj - circle_radius ** 2)
#theta = np.arctan2(jj - circle_radius, ii - circle_radius)
#
## Calculate the normalized coordinates within the original image
#norm_x = (theta / (2 * np.pi)) + 0.5
#norm_y = (r / circle_radius)
#
## Map the normalized coordinates to the range of the original image
#xb = (norm_x * pano.width).astype(int)
#yb = (norm_y * pano.height).astype(int)

#xb = sat.height*(1 - np.arctan( np.sqrt(1 + np.power(((2*sat.height-jj)/(ii)), 2))*ii/c ))
#yb = sat.width*np.arctan((2*sat.height-jj)/ii)
#
#r = c*np.tan((sat.height-ii/(2*np.pi))*np.pi/sat.height)
#theta = 2*np.pi*jj/(sat.width)
#
#xb = center_x + (r * np.cos(theta)).astype(int)
#yb = center_y + (r * np.sin(theta)).astype(int)

#xb = r*np.cos(theta)
#yb = -r*np.sin(theta)

#test = sample_bilinear(pano_arr, xb, yb)

#Image.fromarray(np.uint8(test)).save('test_new.png')
#Image.fromarray(np.uint8(pano_arr)).save('test_pp.png')
#Image.fromarray(np.uint8(sat_arr)).save('test_ss.png')

#print(f'xp: {xp.shape}\nyp: {yp.shape}\nr: {r.shape}\ntheta: {theta.shape}\nxb: {xb.shape}\nyb: {yb.shape}\nii: {ii.shape}\njj: {jj.shape}')
######################################################################333
#from PIL import Image, ImageDraw
#
#def bend_image_around_circle(image_path, circle_radius):
#    # Load the image
#    original_image = Image.open(image_path)
#    original_image = ImageOps.flip(original_image)
#
#    # Calculate the diameter of the circle
#    circle_diameter = circle_radius * 2
#
#    # Create a blank image for the output
#    output_image = Image.new("RGBA", (circle_diameter, circle_diameter), (0, 0, 0, 0))
#
#    # Create a drawing object
#    draw = ImageDraw.Draw(output_image)
#
#    # Iterate over each pixel in the output image
#    c = 100
#    H = sat.height
#    W = sat.width
#
#    for x in range(circle_diameter):
#        for y in range(circle_diameter):
#            # Calculate the relative position within the circle
#            rel_x = x - circle_radius
#            rel_y = y - circle_radius
#
#            # Calculate the polar coordinates
#            #r = c * math.tan((1000-rel_x)*math.pi/1000)
#            #theta = 2*math.pi*rel_y/500
#            r = math.sqrt(rel_x ** 2 + rel_y ** 2)
#            theta = math.atan2(rel_y, rel_x)
#
#            # Calculate the normalized coordinates within the original image
#            norm_x = (theta / (2 * math.pi)) + 0.5
#            norm_y = (r / circle_radius)
#
#            # Map the normalized coordinates to the range of the original image
#            source_x = int(norm_x * original_image.width)
#            source_y = int(norm_y * original_image.height)
#
#            # Check if the source coordinates are within the boundaries of the original image
#            if 0 <= source_x < original_image.width and 0 <= source_y < original_image.height:
#                # Get the pixel color from the original image
#                pixel = original_image.getpixel((source_x, source_y))
#            else:
#                # Use a transparent pixel if the source coordinates are outside the original image boundaries
#                pixel = (0, 0, 0, 0)
#
#            # Draw the pixel in the output image
#            draw.point((x, y), fill=pixel)
#
#    output_image.save('test.png')
#
## Example usage
#image_path = "/gpfs3/scratch/xzhang31/VIGOR/NewYork/panorama/rTW64elYRVtD5DWJ9kBgnA,40.731011,-73.995289,.jpg"
#circle_radius = 500
#bend_image_around_circle(image_path, circle_radius)
#
#def custom_distortion_mapping(x, y, width, height, inner_strength, outer_strength):
#    cx = width / 2
#    cy = height / 2
#    
#    # Calculate distance from the center
#    dx = x - cx
#    dy = y - cy
#    distance = np.sqrt(dx*dx + dy*dy)
#    
#    # Calculate scaling factor based on distance from the center
#    scaling_factor = 1.0 + ((distance / cx) * outer_strength)
#    
#    # Apply perspective transformation on the inner part
#    if distance < cx:
#        scaling_factor = 1.0 - ((distance / cx) * inner_strength)
#    
#    # Calculate new coordinates
#    new_x = cx + dx * scaling_factor
#    new_y = cy + dy * scaling_factor
#    
#    return int(new_x), int(new_y)
#
#def apply_custom_distortion(image, inner_strength, outer_strength):
#    width, height = image.size
#    distorted_image = Image.new("RGB", (width, height))
#    
#    for y in range(height):
#        for x in range(width):
#            src_x, src_y = custom_distortion_mapping(x, y, width, height, inner_strength, outer_strength)
#            if 0 <= src_x < width and 0 <= src_y < height:
#                pixel_value = image.getpixel((src_x, src_y))
#                distorted_image.putpixel((x, y), pixel_value)
#    
#    return distorted_image
#
## Load the fish eye image
#fisheye_image = Image.open("test.png")
#
## Set the distortion strengths (experiment with different values)
#inner_distortion_strength = 0.5
#outer_distortion_strength = 0.5
#
## Apply the custom distortion
#distorted_image = apply_custom_distortion(fisheye_image, inner_distortion_strength, outer_distortion_strength)
#
## Save the distorted image
#distorted_image.save("test2.png")



