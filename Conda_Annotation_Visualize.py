#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from os import listdir
from PIL import Image, ImageDraw
import xml.etree.ElementTree as ET 
import xmltodict
import matplotlib.pyplot as plt
import matplotlib.image as img
import glob
import os

from tqdm import tqdm
import pyvips
import pylibczi
from pylibczi import CziScene
import czifile
from czifile import CziFile 
# import pyvips as vips

import vips_utils


""" Convert np array to vips """
dtype_to_format = {
    'uint8': 'uchar',
    'int8': 'char',
    'uint16': 'ushort',
    'int16': 'short',
    'uint32': 'uint',
    'int32': 'int',
    'float32': 'float',
    'float64': 'double',
    'complex64': 'complex',
    'complex128': 'dpcomplex'
}

def numpy2vips(a):
    height, width, bands = a.shape
    linear = a.reshape(width * height * bands)
    vi = pyvips.Image.new_from_memory(linear.data, width, height, bands,
                                      dtype_to_format[str(a.dtype)])
    return vi

CZI_PATH = '../data/Batch_01_Original_CZI_files/'
CZ_PATH = '../data/Uploads_with_Corrections/'


# In[2]:


def extract_rectangles_cz(PATH, metadata_xml):
    """
    Method to extract the coordinates of rectangles drawn on the image 
    
    input: PATH and metadata name 
    
    
    return: 
    rectangles: dictionnary
    
    {
    'rectangle_1': [Left, Top, Width, Height],
    'rectangle_2': [...],
    
    ...}
    
    """
    
    rectangles = {}
    
    metadata_name = PATH + metadata_xml
    doc = xmltodict.parse(open(metadata_name, 'r', encoding='utf-8').read())
    dic_rectangle = doc['GraphicsDocument']['Elements']['Rectangle']

    for rect in range(0, len(dic_rectangle)):
        geometry = dic_rectangle[rect]['Geometry']
        rectangles['rectangle_{}'.format(rect)] = [geometry['Left'], geometry['Top'], geometry['Width'], geometry['Height']]
        
    return(rectangles)

def extract_annotations_coordinates_cz(PATH, metadata_xml):
    """
    Method to extract the annotations (points) drawn on the image
    
    input: PATH and metadata name 
    
    return: dictionnary of annotations (X,Y)
    
    {
    'annotation_1': [X_1, Y_1], 
    'annotation_1': [X_2, Y_2],
    ...
    }
    
    """
    
    annotations = {}
    
    metadata_name = PATH + metadata_xml
    doc = xmltodict.parse(open(metadata_name, 'r', encoding='utf-8').read())
    dic_marker = doc['GraphicsDocument']['Elements']['Marker']

    for coordinate in range(0, len(dic_marker)):
        geometry = dic_marker[coordinate]['Geometry']
        annotations['annotation_{}'.format(coordinate)] = [int(float(geometry['X'])), int(float(geometry['Y']))]

    return(annotations)

def extract_scaling_factor_cz(PATH, metadata_xml):
    """
    Extract the scaling factor 
    
    input: PATH and metadata name 
    return: float 
    """
    
    metadata_name = PATH + metadata_xml
    doc = xmltodict.parse(open(metadata_name, 'r', encoding='utf-8').read())

    scaling_factor = doc['GraphicsDocument']['Scaling']['Items']['Distance'][0]['Value']
    scaling_factor = float(scaling_factor)
    
    return(scaling_factor)


# In[3]:


filenames = ['1-102-Temporal_AT8.czi',
 '1-154-Temporal_AT8.czi',
#  '1-209-Temporal_AT8.czi',
#  '1-225-Temporal_AT8.czi',
#  '1-254-Temporal_AT8.czi',
 '1-271-Temporal_AT8.czi',
#  '1-290-Temporal_AT8.czi',
 '1-297-Temporal_AT8.czi',
 '1-466-Temporal_AT8.czi',
 '1-516-Temporal_AT8.czi',
#  '1-573-Temporal_AT8.czi',
 '1-621-Temporal_AT8.czi',
 '1-693-Temporal_AT8.czi',
 '1-695-Temporal_AT8.czi',
 '1-717-Temporal_AT8.czi',
#  '1-751-Temporal_AT8.czi',
#  '1-755-Temporal_AT8.czi',
 '1-756-Temporal_AT8.czi',
#  '1-838-Temporal_AT8.czi',
 '1-907-Temporal_AT8.czi',
 '1-923-Temporal_AT8.czi',
 '1-960-Temporal_AT8.czi']


# In[4]:


filenames = filenames[12:]
filenames


# In[ ]:


for filename in tqdm(filenames):
#     filename = filenames[i]
    name = filename.split('.')[0]
    img = czifile.imread(CZI_PATH + filename)
    
    numpy_array = np.array(img, dtype = np.uint8)
    scenes = numpy_array.shape[0]
    time = numpy_array.shape[1]
    height = numpy_array.shape[2]
    width = numpy_array.shape[3]
    channels = numpy_array.shape[4]
    numpy_array_reshaped = numpy_array.reshape((height, width, channels))
    print(numpy_array_reshaped.shape) 
    vips = numpy2vips(numpy_array_reshaped) 
    
    vips_resized = vips.resize(0.1)
    width = vips_resized.width
    height = vips_resized.height
    print('width:', width, 'height:', height)

    plt.rcParams['savefig.dpi'] = 800
    plt.rcParams['figure.dpi'] = 800
    plt.axis('off')
    vips_utils.show_vips(vips_resized, show=False)
    full_address = './Visualization/' + name + '.png'
    plt.savefig(full_address, dpi=400)
    del vips_resized, full_address 
    print('original done')
    
    vips_resized = vips.resize(0.1)
    metadata_name = name + '.cz'
    rectangles = extract_rectangles_cz(CZ_PATH, metadata_name)
    annotations = extract_annotations_coordinates_cz(CZ_PATH, metadata_name)
    scaling_factor = 0.1
    color = [255,0,0]
    for i in range(0, len(rectangles)):
        coord = rectangles['rectangle_{}'.format(i)]
        left = int(float(coord[0])) * scaling_factor
        top = int(float(coord[1])) * scaling_factor
        width = int(float(coord[2])) * scaling_factor
        height = int(float(coord[3])) * scaling_factor
        vips_resized = vips_resized.draw_rect(color, left, top, width, height, fill=True)
    plt.axis('off')
    vips_utils.show_vips(vips_resized, show=False)
    full_address = './Visualization/' + name + '_rec.png'
    plt.savefig(full_address, dpi=400)
    del vips_resized, full_address
    print('rectangles done!')
    
    vips_resized = vips.resize(0.1)
    for i in range(0, len(annotations)):
        coord = annotations['annotation_{}'.format(i)]
        x = coord[0]*scaling_factor
        y = coord[1]*scaling_factor
        #print(x, y)
        vips_resized = vips_resized.draw_circle(color, x, y, 50, fill=True)
    plt.axis('off')
    vips_utils.show_vips(vips_resized, show=False)
    full_address = './Visualization/' + name + '_coo.png'
    plt.savefig(full_address, dpi=400)
    del vips_resized, full_address 
    print('coordinate done!')
    del vips 
    
#     scaling_factor = extract_scaling_factor_cz(CZ_PATH, metadata_name)
    


# In[ ]:





# In[14]:


filenames = ['1-271-Temporal_AT8.czi']
for filename in tqdm(filenames):
#     filename = filenames[i]
    name = filename.split('.')[0]
    img = czifile.imread(CZI_PATH + filename)
    
    numpy_array = np.array(img, dtype = np.uint8)
    scenes = numpy_array.shape[0]
    time = numpy_array.shape[1]
    height = numpy_array.shape[2]
    width = numpy_array.shape[3]
    channels = numpy_array.shape[4]
    numpy_array_reshaped = numpy_array.reshape((height, width, channels))
    print(numpy_array_reshaped.shape) 
    vips = numpy2vips(numpy_array_reshaped) 
    
    vips_resized = vips.resize(0.1)
    width = vips_resized.width
    height = vips_resized.height
    print('width:', width, 'height:', height)

    plt.rcParams['savefig.dpi'] = 800
    plt.rcParams['figure.dpi'] = 800
    plt.axis('off')
    vips_utils.show_vips(vips_resized)


    


# In[15]:


'''
Cr
No need to run 
'''
metadata_name = name + '.cz'
rectangles = extract_rectangles_cz(CZ_PATH, metadata_name)
annotations = extract_annotations_coordinates_cz(CZ_PATH, metadata_name)
scaling_factor = 0.1
color = [255,0,0]
coord = rectangles['rectangle_{}'.format(0)]
left = int(float(coord[0])) * scaling_factor
top = int(float(coord[1])) * scaling_factor
width = int(float(coord[2])) * scaling_factor
height = int(float(coord[3])) * scaling_factor

#cropped_img = vips.crop(left, top, width, height)
cropped_img = vips_resized.extract_area(left, top, width, height) # similar to crop

vips_utils.show_vips(cropped_img)


# In[18]:


# vips_resized = vips.resize(0.1)
metadata_name = name + '.cz'
rectangles = extract_rectangles_cz(CZ_PATH, metadata_name)
annotations = extract_annotations_coordinates_cz(CZ_PATH, metadata_name)
scaling_factor = 0.1
color = [255,0,0]


for i in range(0, len(annotations)):
    coord = annotations['annotation_{}'.format(i)]
    x = coord[0]*scaling_factor
    y = coord[1]*scaling_factor
    #print(x, y)
    vips_resized = vips_resized.draw_circle(color, x, y, 20, fill=False)
#     vips_resized = vips_resized.draw_circle(color, x, y, 6, fill=True)
plt.axis('off')
vips_utils.show_vips(vips_resized)


# In[21]:


# for i in range(0, len(rectangles)):
# coord = rectangles['rectangle_{}'.format(i)]
coord = rectangles['rectangle_{}'.format(2)]
left = int(float(coord[0])) * scaling_factor
top = int(float(coord[1])) * scaling_factor
width = int(float(coord[2])) * scaling_factor
height = int(float(coord[3])) * scaling_factor

#cropped_img = vips.crop(left, top, width, height)
cropped_img = vips_resized.extract_area(left, top, width, height) # similar to crop

vips_utils.show_vips(cropped_img)


# In[ ]:




