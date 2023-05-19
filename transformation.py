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

cropped_pano = pano.crop((0, int(pano.size[1]*0.5), pano.size[0], pano.size[1]))
pano_arr = np.array(pano)
sat_arr = np.array(sat)

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
#ii=ii+0.001
#
#center_x, center_y = pano.width // 2, pano.height // 2
#
#xb = sat.height*(1 - np.arctan( np.sqrt(1 + np.power(((2*sat.height-jj)/(ii)), 2))*ii/c ))
#yb = sat.width*np.arctan((2*sat.height-jj)/ii)

#xb = sat.height*(1 - np.arctan( np.sqrt(1 + np.power(((2*sat.height-jj)/(ii)), 2))*ii/c ))
#yb = sat.width*np.arctan((2*sat.height-jj)/ii)

#r = c*np.tan((sat.height-ii/(2*np.pi))*np.pi/sat.height)
#theta = 2*np.pi*jj/(sat.width)
#
#xb = center_x + (r * np.cos(theta)).astype(int)
#yb = center_y + (r * np.sin(theta)).astype(int)

#xb = r*np.cos(theta/2)
#yb = -r*np.sin(theta/2)

#test = sample_bilinear(pano_arr, xb, yb)
#
#Image.fromarray(np.uint8(test)).save('test.png')
#Image.fromarray(np.uint8(pano_arr)).save('test_pp.png')
#Image.fromarray(np.uint8(sat_arr)).save('test_ss.png')
#
#print(f'xp: {xp.shape}\nyp: {yp.shape}\nr: {r.shape}\ntheta: {theta.shape}\nxb: {xb.shape}\nyb: {yb.shape}\nii: {ii.shape}\njj: {jj.shape}')
#
# print(pano.size) (width, height)
#height = sat.size[0]   # Height of polar transformed aerial image
#width = sat.size[1]    # Width of polar transformed aerial image
#c = 100 

#i = np.arange(0, height)
#j = np.arange(0, width)
#jj, ii = np.meshgrid(j, i)

#y = (width*np.arctan((height-ii)/jj))/(2*np.pi)
#x = height*(1-(np.arctan(np.sqrt(1+np.power((height-ii)/jj, 2))*jj/c))/np.pi)

#y = S/2. - S/2./height*(height-1-ii)*np.sin(2*np.pi*jj/width)
#x = S/2. + S/2./height*(height-1-ii)*np.cos(2*np.pi*jj/width)

#print(f'jj\n{jj}\n{jj.shape}\nii\n{ii}\n{ii.shape}')
#print(f'x\n{x}\n{x.shape}\ny\n{y}\n{y.shape}')

#test = sample_bilinear(pano_arr, x, y)
#Image.fromarray(np.uint8(test)).save('test.png')
#Image.fromarray(np.uint8(pano_arr)).save('test_pp.png')
#Image.fromarray(np.uint8(sat_arr)).save('test_ss.png')

######################################
#import numpy as np
#
#c_street = [0]
#c_building = [100,150,200,250,300,350,400,450,500,550]
#
#for c1 in c_street:
#    for c2 in c_building:
#        # Assume H, W, c are defined as the height, width, and a constant respectively.
#        H, W, c = sat.height, sat.width, 50  # example values
#
#        boundary = H // 2
#        # Assume we have a panoramic image 'pano_img' of shape (H, W)
#        # We need to create a blank bird's-eye view image 'bird_eye_img'
#        bird_eye_img = np.zeros_like(sat_arr)
#        center_x, center_y = W // 2, H // 2
#        print('ln', '*'*8, bird_eye_img.shape, pano_arr. shape)
#        # Loop over each pixel in the panoramic image.
#        for xp in range(H):
#            for yp in range(W):
#                if xp < boundary:
#                    c = c1
#                else: c = c2
#                # Apply the transformations.
#                r_co = c * np.tan((H - xp) * np.pi / H)
#                theta = 2 * np.pi * yp / W
#                # Calculate the corresponding pixels in bird's-eye view image
#                xb = center_x + int(r_co * np.cos(theta))
#                yb = center_y + int(r_co * np.sin(theta))
#                # Check if the calculated indices are within the image dimensions.
#                if 0 <= xb < W and 0 <= yb < H:
#                    bird_eye_img[yb, xb] = pano_arr[xp, yp]
#
#        Image.fromarray(np.uint8(bird_eye_img)).save(f'st_{c1}_buil_{c2}_test.png')
#Image.fromarray(np.uint8(pano_arr)).save('test_pp.png')
#Image.fromarray(np.uint8(sat_arr)).save('test_ss.png')

from PIL import Image, ImageDraw

def bend_image_around_circle(image_path, circle_radius):
    # Load the image
    original_image = Image.open(image_path)
    original_image = ImageOps.flip(original_image)

    # Calculate the diameter of the circle
    circle_diameter = circle_radius * 2

    # Create a blank image for the output
    output_image = Image.new("RGBA", (circle_diameter, circle_diameter), (0, 0, 0, 0))

    # Create a drawing object
    draw = ImageDraw.Draw(output_image)

    # Iterate over each pixel in the output image
    c = 100
    H = sat.height
    W = sat.width

    for x in range(circle_diameter):
        for y in range(circle_diameter):
            # Calculate the relative position within the circle
            rel_x = x - circle_radius
            rel_y = y - circle_radius

            # Calculate the polar coordinates
            #r = c * math.tan((1000-rel_x)*math.pi/1000)
            #theta = 2*math.pi*rel_y/500
            r = math.sqrt(rel_x ** 2 + rel_y ** 2)
            theta = math.atan2(rel_y, rel_x)

            # Calculate the normalized coordinates within the original image
            norm_x = (theta / (2 * math.pi)) + 0.5
            norm_y = (r / circle_radius)

            # Map the normalized coordinates to the range of the original image
            source_x = int(norm_x * original_image.width)
            source_y = int(norm_y * original_image.height)

            # Check if the source coordinates are within the boundaries of the original image
            if 0 <= source_x < original_image.width and 0 <= source_y < original_image.height:
                # Get the pixel color from the original image
                pixel = original_image.getpixel((source_x, source_y))
            else:
                # Use a transparent pixel if the source coordinates are outside the original image boundaries
                pixel = (0, 0, 0, 0)

            # Draw the pixel in the output image
            draw.point((x, y), fill=pixel)

    output_image.save('test.png')

# Example usage
image_path = "/gpfs3/scratch/xzhang31/VIGOR/NewYork/panorama/rTW64elYRVtD5DWJ9kBgnA,40.731011,-73.995289,.jpg"
circle_radius = 500
bend_image_around_circle(image_path, circle_radius)

from PIL import Image
import numpy as np

def custom_distortion_mapping(x, y, width, height, inner_strength, outer_strength):
    cx = width / 2
    cy = height / 2
    
    # Calculate distance from the center
    dx = x - cx
    dy = y - cy
    distance = np.sqrt(dx*dx + dy*dy)
    
    # Calculate scaling factor based on distance from the center
    scaling_factor = 1.0 + ((distance / cx) * outer_strength)
    
    # Apply perspective transformation on the inner part
    if distance < cx:
        scaling_factor = 1.0 - ((distance / cx) * inner_strength)
    
    # Calculate new coordinates
    new_x = cx + dx * scaling_factor
    new_y = cy + dy * scaling_factor
    
    return int(new_x), int(new_y)

def apply_custom_distortion(image, inner_strength, outer_strength):
    width, height = image.size
    distorted_image = Image.new("RGB", (width, height))
    
    for y in range(height):
        for x in range(width):
            src_x, src_y = custom_distortion_mapping(x, y, width, height, inner_strength, outer_strength)
            if 0 <= src_x < width and 0 <= src_y < height:
                pixel_value = image.getpixel((src_x, src_y))
                distorted_image.putpixel((x, y), pixel_value)
    
    return distorted_image

# Load the fish eye image
fisheye_image = Image.open("test.png")

# Set the distortion strengths (experiment with different values)
inner_distortion_strength = 0.5
outer_distortion_strength = 0.5

# Apply the custom distortion
distorted_image = apply_custom_distortion(fisheye_image, inner_distortion_strength, outer_distortion_strength)

# Save the distorted image
distorted_image.save("test2.png")



