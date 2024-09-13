import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import arcpy
import os
from arcpy import env
from arcpy.sa import *
# Input data source
inFolder = r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\PDSI'#set the worksapce where NetCDF exist
arcpy.env.workspace = inFolder
arcpy.env.overwriteOutput = True  #If the file has already existed, it'll overwrite.

# Set output folder
OutputFolder = r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\PDSI\111'
# Loop through a list of files in the workspace

arcpy.CheckOutExtension("spatial")

variable = 'PDSI'
in_folder,ncfile=os.path.split(inFolder)

NCfiles = [nc_file for nc_file in os.listdir(inFolder) if nc_file.endswith(".nc")]

for nc_file1 in NCfiles:
    print(nc_file1)
    nc_FP = arcpy.NetCDFFileProperties(inFolder+'\\'+nc_file1)
    nc_Dim = nc_FP.getDimensions()
    top = nc_FP.getDimensionSize("time")
    #code=nc_file1[16:27]
    dimension=nc_Dim[2]
    #year_str = nc_file1.split(".")[4]

    for i in range(0, 12):
        print i
        dimension_values = nc_FP.getDimensionValue(dimension, i)

        doyy=dimension_values

        year_str = doyy.split('-')[0]
        month_str = doyy.split('-')[1]
        #if int(month_str)<10:
        #   month_str="0"+month_str
        fileroot = 'PDSI_'+year_str+'_'+month_str
        dv1 = ['time', dimension_values]
        dimension_values = [dv1]

        outRaster = OutputFolder + '\\' + fileroot
        arcpy.MakeNetCDFRasterLayer_md(nc_file1, variable, 'lon', 'lat', outRaster,'',dimension_values, '')
        arcpy.CopyRaster_management(outRaster, outRaster+'.tif', "", "", "", "NONE", "NONE", "")
