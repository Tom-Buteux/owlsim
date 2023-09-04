# imports 
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from scipy.spatial import cKDTree
from astropy.convolution import Gaussian2DKernel
from astropy.convolution import convolve


# functions
def find_orthogonal_set(vec):
    """Find two vectors that are orthogonal to the given vector `vec`."""
    if np.linalg.norm(vec) == 0:
        raise ValueError("The input vector must not be the zero vector.")
    
    # Normalize the input vector
    vec = vec / np.linalg.norm(vec)
    
    # Initialize candidates for orthogonal vectors
    candidates = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    
    # Make sure the candidates are not parallel to the input vector
    candidates = [c for c in candidates if np.abs(np.dot(vec, c)) < 1.0]
    
    # Choose one candidate and find a vector orthogonal to it and the input vector
    chosen = candidates[0]
    ortho1 = np.cross(vec, chosen)
    ortho1 = ortho1 / np.linalg.norm(ortho1)  # normalize
    
    # Find another vector that is orthogonal to both `vec` and `ortho1`
    ortho2 = np.cross(vec, ortho1)
    ortho2 = ortho2 / np.linalg.norm(ortho2)  # normalize
    
    return ortho1, ortho2

# load in the catalogue for the correct pass band (initial case will be the 2MASS H-band)
# this will be a list of stars with their positions (RA/DEC deg) and magnitudes (Hmag) 
hdul = fits.open('2MASS 12,12, 10 deg.fit')
data = hdul[1].data
data = pd.DataFrame(data)
print('length of catalogue = ', len(data))
print('max RA = ', np.max(data['_RAJ2000']))
print('min RA = ', np.min(data['_RAJ2000']))
print('max DEC = ', np.max(data['_DEJ2000']))
print('min DEC = ', np.min(data['_DEJ2000']))


# for each star in the catalogue, calcuate its estimated flux (F)
# to do this i need the zero point flux (F0) and the magnitude of the star (m)
F0 = 1.133e-13 # W/m^2/micron^2
m = data['Hmag'] # H-band magnitude
F = F0 * 10**(-m/2.5) # flux in W/m^2/micron^2
data['Flux'] = F

energy_per_photon = 1.2e-19 # J

# scale the flux of the stars to the exposure time of the image
# this will be the
#  flux of the star in the image
exposure_time = 5 # seconds
data['Flux'] = data['Flux'] * exposure_time

# for each star in the catalogue, calculate its position its cartesian coordinates on a unit sphere
data['x'] = np.cos(np.radians(data['_DEJ2000'])) * np.cos(np.radians(data['_RAJ2000']))
data['y'] = np.cos(np.radians(data['_DEJ2000'])) * np.sin(np.radians(data['_RAJ2000']))
data['z'] = np.sin(np.radians(data['_DEJ2000']))


# use the centre of the simulated image as the centre of the field of view
# this will be the centre of the image (x,y) = (0,0)
# instrument field of view is 4.4 x 5.5 deg
inst_FOV = [4.4, 5.5] # deg
cen_RA = 180 # deg
cen_DEC = 10 # deg

# calculate the cartesian coordinates of the centre of the field of view
cen_x = np.cos(np.radians(cen_DEC)) * np.cos(np.radians(cen_RA))
cen_y = np.cos(np.radians(cen_DEC)) * np.sin(np.radians(cen_RA))
cen_z = np.sin(np.radians(cen_DEC))

# calculate the cartesian distance that relates to the angular distance of the field of view
# this will be the radius of the field of view
# from centre to corner in degrees 
cen_to_corner = np.sqrt((inst_FOV[0]/2)**2 + (inst_FOV[1]/2)**2)
# from centre to corner in radians
cen_to_corner = np.radians(cen_to_corner)


return_FOV = 2 * np.sin(cen_to_corner/2)
print('return_FOV = ', return_FOV)

# create a cartesian tree of the stars in the catalogue
cartesian_tree = cKDTree(data[['x', 'y', 'z']])

# query the tree for all stars within the field of view
# this will return the indices of the stars in the catalogue that are within the field of view
neighbors = cartesian_tree.query_ball_point([cen_x, cen_y, cen_z], r= return_FOV)

# create a new dataframe of the stars that are within the field of view
# this will be the stars that are in the image
image = data.iloc[neighbors]
# reset the index of the image dataframe
image = image.reset_index(drop=True)
print('length of image = ', len(image))

# camera 2D plane 
plane_normal = [cen_x, cen_y, cen_z]
# find the orthogonal set of the plane normal
u,v = find_orthogonal_set(plane_normal)


# Your vectorized function
def project_points_to_plane(points, n):
    vecs = points - np.array([0, 0, 0])
    d = np.dot(vecs, n)
    projected_points = points - d[:, np.newaxis] * n
    return projected_points

# Assuming image is a DataFrame with 'x', 'y', 'z' columns
points = image[['x', 'y', 'z']].to_numpy()
plane_normal = np.array([cen_x, cen_y, cen_z])  

# Use the vectorized function
projected_points = project_points_to_plane(points, plane_normal)
on_plane = [[np.dot(point, u), np.dot(point, v)] for point in projected_points] 


# place the stars in the image onto the camera plane
# this will be the 2D coordinates of the stars in the image in x and y pixels

# find the extent of the image in x and y
dist_x = 2 * np.sin(np.radians(np.max(inst_FOV[1]))/2)
dist_y = 2 * np.sin(np.radians(np.max(inst_FOV[0]))/2)

# convert on_plane into a dataframe
on_plane = pd.DataFrame(on_plane, columns=['x', 'y'])

# filter out the stars that are outside the field of view
on_plane = on_plane[(on_plane['x'] >= -dist_x/2) & (on_plane['x'] <= dist_x/2) & (on_plane['y'] >= -dist_y/2) & (on_plane['y'] <= dist_y/2)]

# add the flux of the stars to the dataframe
on_plane['Flux'] = image['Flux']
# convert the flux values to interger values between 0 and 255
on_plane['count'] = on_plane['Flux'] / energy_per_photon
on_plane['count'] = on_plane['count'].round(0).astype(int)

# as the imager is 14 bit the max pixel value is 16383
# threshold the pixel values above 16383 to 16383
# print the length of the dataframe with count values above 16383
print('number of thresholded pixels = ', len(on_plane[on_plane['count'] > 16383]))
on_plane['count'][on_plane['count'] > 16383] = 16383





# add half the dist of the image to the x and y coordinates
on_plane['x'] = on_plane['x'] + dist_x/2
on_plane['y'] = on_plane['y'] + dist_y/2
# convert the x and y coordinates into pixels
on_plane['x'] = on_plane['x'] * 639 / np.max(on_plane['x'])
on_plane['y'] = on_plane['y'] * 511 / np.max(on_plane['y'])
# round the x and y coordinates to the nearest integer
on_plane['x'] = on_plane['x'].round(0).astype(int)
on_plane['y'] = on_plane['y'].round(0).astype(int)

# print the min and max x and y coordinates
print('max x = ', np.max(on_plane['x']))
print('min x = ', np.min(on_plane['x']))
print('max y = ', np.max(on_plane['y']))
print('min y = ', np.min(on_plane['y']))

# create an image by placing the stars onto the camera plane
# the x and y coordinates of the stars will be the pixels in the image, the flux will be the value of the pixel
# this will be the image
image = np.zeros((512, 640))
# reset the index of the on_plane dataframe
on_plane = on_plane.reset_index(drop=True)


for i in range(len(on_plane)):
    image[on_plane['y'][i], on_plane['x'][i]] += on_plane['count'][i]

# apply a psf to the image
# this will be the psf
psf = Gaussian2DKernel(2)
# convolve the image with the psf
image = convolve(image, psf)

# convert the image to 14-bit
image = image / np.max(image) * 16383
image = image.round(0).astype(int)


# print the min and max pixel values of the image
print('max pixel value = ', np.max(image))  
print('min pixel value = ', np.min(image))








plt.imshow(image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('test_owlsim.png', dpi=300)
plt.show()

# save the image as a fits file
hdu = fits.PrimaryHDU(image)

# set the reference pixel to the centre of the image
hdu.header['CRPIX1'] = 320
hdu.header['CRPIX2'] = 256
# set the reference pixel to the centre of the field of view
hdu.header['CRVAL1'] = cen_RA
hdu.header['CRVAL2'] = cen_DEC
# set the pixel scale
hdu.header['CDELT1'] = -0.006875
hdu.header['CDELT2'] = 0.006875
# set the units
hdu.header['CUNIT1'] = 'deg'
hdu.header['CUNIT2'] = 'deg'
# set the projection type
hdu.header['CTYPE1'] = 'RA---TAN'
hdu.header['CTYPE2'] = 'DEC--TAN'

hdu.writeto('test_owlsim.fits', overwrite=True)








"""

# Create an empty image (Owl has a 512x640 CCD)
image = np.zeros((512, 640))



# Scale the PSF by the star's flux (say the flux is 1000)
flux = 1000
scaled_psf = psf * flux

# Place the star at the center of the image
image[4:7, 4:7] += scaled_psf

# Add noise (for simplicity, just using random Gaussian noise here)
noise = np.random.normal(0, 10, image.shape)  # 0 mean, 10 standard deviation
image += noise

# Apply saturation limit (say 65535 for a 16-bit camera)
image[image > 65535] = 65535
"""
