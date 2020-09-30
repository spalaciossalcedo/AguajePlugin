
import os
from osgeo import gdal, ogr
import numpy as np


def rgb_mask(input_image, mask_image, out_folder):
    
    """ Apply the classification mask to the Input image.
  
    Arguments:
    input_image -str
      Image to be classified
    mask_image - str
      Mask classified generated with the cnn
    out_folder  
      Directory to place output images into."""

    #input_file_list = [input_image]#['data/landscape7_dem_subset.tif'] #Load the image to mask 
    #input_mask = [mask_image] #Load mask from the previous trained

    rgb = gdal.Open(input_image,gdal.GA_ReadOnly) 

    mask = gdal.Open(mask_image,gdal.GA_ReadOnly)

    if (rgb.RasterCount == 1):
        nbands = 1
    else :
        nbands = 3

    rgb_array = np.zeros((rgb.RasterYSize,rgb.RasterXSize,nbands))
    mask_array = np.zeros((mask.RasterYSize,mask.RasterXSize,nbands))

    mask_array = mask.GetRasterBand(1).ReadAsArray()
    mask_array[mask_array == -9999] = 0

    for n in range(0,nbands):
        rgb_array[:,:,n] = rgb.GetRasterBand(n+1).ReadAsArray()
        rgb_array[:,:,n] = rgb_array[:,:,n] * mask_array


    output_tif_file = os.path.join(out_folder,os.path.basename(input_image).split('.')[0]  + '_image_classified.tif')

    driver = gdal.GetDriverByName('GTiff')
    outrgb = driver.Create(output_tif_file,rgb_array.shape[1],rgb_array.shape[0],nbands,gdal.GDT_Float32)
    outrgb.SetProjection(rgb.GetProjection())
    outrgb.SetGeoTransform(rgb.GetGeoTransform())

    for n in range(0,nbands):
        outrgb.GetRasterBand(n+1).WriteArray(rgb_array[:,:,n],0,0)
    del outrgb

    rgb = None
    mask = None

