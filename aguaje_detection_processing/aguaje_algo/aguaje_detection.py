import os
import inspect
from osgeo import gdal, ogr
import warnings

import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras.backend as kb
sys.stderr = stderr

from ..aguaje_algo import train_model
from ..aguaje_algo import apply_model
from ..aguaje_algo import image_processing
warnings.filterwarnings('ignore')


cmd_folder = os.path.split(inspect.getfile(inspect.currentframe()))[0]

pluginPath = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,'trained_models'))
        
def apply_aguaje(INPUT_RASTER, OUTPUT_RASTER):

    

    window_radius = 256
    internal_window_radius = int(round(window_radius*0.75))
    model = train_model.load_trained_model('Mauritia_38_G',window_radius,internal_window_radius,pluginPath,weighted=True,verbose=False)

    out_folder = os.path.dirname(OUTPUT_RASTER)
    green_tif_file = os.path.join(out_folder,os.path.basename(INPUT_RASTER).split('.')[0]  + '_G.tif')

    if os.path.exists(green_tif_file): #Checking if the green band already exists.
        pass
    else:
        rgb = gdal.Open(INPUT_RASTER,gdal.GA_ReadOnly)
        out_ds = gdal.Translate(green_tif_file, rgb, format='GTiff', bandList=[2])#Loading only green band
        out_ds=None

    feature_file_list = [green_tif_file]
    


    apply_model.apply_semantic_segmentation(feature_file_list,
                                        out_folder,
                                        model,
                                        window_radius,
                                        internal_window_radius=internal_window_radius,
                                        make_tif=True,
                                        make_png=False,
                                        local_scale_flag='mean')

    name_saved = apply_model.convert_probabilities_to_classes(feature_file_list,
                                                              out_folder,
                                                              response_thresholds=['background',0.7],#0.99
                                                              response_order=[0,1],
                                                              feature_band_to_plot=0)                                   


    out_imag = os.path.join(out_folder,name_saved)
    os.rename(out_imag, OUTPUT_RASTER)
    os.remove(green_tif_file) #delete the green image
    image_processing.rgb_mask(INPUT_RASTER,OUTPUT_RASTER,out_folder)
    kb.clear_session()
    return OUTPUT_RASTER