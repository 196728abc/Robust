import time

import arcpy
import os
from arcpy import env
from arcpy.sa import *
# Input data source
inFolder = r'C:\Users\lijia\Desktop\PAR'#set the worksapce where NetCDF exist
arcpy.env.workspace = inFolder
arcpy.env.overwriteOutput = True  #If the file has already existed, it'll overwrite.

# Set output folder
OutputFolder = 'C:/Users/lijia/Desktop/PAR/direct PAR/'
# Loop through a list of files in the workspace

# NCfiles = arcpy.ListFiles("*.nc")
NCfiles = [nc_file for nc_file in os.listdir(inFolder) if nc_file.endswith(".nc4")]
nums=len(NCfiles)
#for num,filename in enumerate(NCfiles):


for j in range(3*nums/4,nums):
    filename=NCfiles[j]
    inNCfile = arcpy.env.workspace + "\\" + filename
    fileroot = 'P_' + filename[27:31]+'_'+filename[31:33]
    TempLayerFile = "PS"
    outRaster = OutputFolder + '/' + fileroot
    # Process: Make NetCDF Raster Layer
    arcpy.MakeNetCDFRasterLayer_md(inNCfile, "PS", "lon", "lat", TempLayerFile, "", "", "")
    arcpy.Resample_management(TempLayerFile, 'Tem_cor', "0.5", "CUBIC")
    arcpy.CopyRaster_management('Tem_cor', outRaster + ".tif", format='TIFF')
    print (filename+r"  has successfully been processed")


print r'all have successfully been processed'
print arcpy.GetMessages()
