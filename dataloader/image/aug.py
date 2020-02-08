from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from scipy import ndimage
import skimage 
import numpy as np
from skimage.transform import rotate, AffineTransform, warp

def rotate(image, mask, rang):
	value = randint(rang[0], rang[1])
	return rotate(image, angle=value, mode = 'wrap'),rotate(mask, angle=value, mode = 'wrap')

def wrapShift(image, mask, rang):
	transform = AffineTransform(translation=(rang[0],rang[1]))
	return warp(image,transform,mode='wrap'), warp(mask,transform,mode='wrap')

def flipR(image, mask):
	return np.fliplr(image), np.fliplr(mask)

def flipV(image, mask):
	return np.fliplud(image), np.fliplud(mask)

def Noise(image, mask, sigma):
	return random_noise(image,var=sigma**2), random_noise(mask,var=sigma**2)

def Blur(image, mask, sigma = 1, out_channel = True):
	return gaussian(image,sigma=sigma,multichannel=True), gaussian(mask,sigma=sigma,multichannel=out_channel)

