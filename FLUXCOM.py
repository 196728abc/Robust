import numpy as np
import netCDF4 as nc4
from netCDF4 import Dataset
import arcpy
from arcpy import env
from arcpy.sa import *
import math
import copy
from datetime import datetime

'''
env.workspace='D:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/landCover'
inRaster='MCD12C1.A2015001.006.2018053185652.hdf'
inSQLClause='VALUE=1'    # landCover setting  0_water, 1:ENF, 2:EBF, 3:DNF, 4:DBF, 5:MF, 6:Closed shrubland, 7:open shrubland, 8:woody savannas; 9:savannas 10:grasslands; 11:permanent wetlands, 12:croplands, 13:urban and bulit-up, 14:cropland/natural vegetatic, 15:snow and ice, 16:barren or sparsely vegetated
arcpy.CheckOutExtension('Spatial')
attExtract=ExtractByAttributes(inRaster,inSQLClause)
# attExtract.save('D:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/landCover/land_cover_XX')
'''


arcpy.env.workspace=r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\FLUXCOM\CERES_GPCP_2001-2014\DT_MLM_MARS'
NCfiles=arcpy.ListFiles('*.nc')
print(NCfiles)

datelst = np.arange(len(NCfiles))

data=np.zeros((len(NCfiles),2))
n=0
for i in NCfiles:
    meanResult=arcpy.GetRasterProperties_management(i,'MEAN')
    minRes=meanResult.getOutput(0)
    data[n,0]=datelst[n]
    data[n,1]=minRes
    n=n+1