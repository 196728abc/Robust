# encoding: utf-8
import arcpy
from arcpy import env
from arcpy.sa import *
import os
import glob

GPP_SIF=glob.glob(os.path.join(r'C:\Users\lijia\Desktop\1111\new folder','*.tif'))
output_path1 = "C:/Users/lijia/Desktop/1111/new folder/"

arcpy.env.workspace = output_path1
arcpy.env.overwriteOutput = True

FLUXCOM_GOME2=GPP_SIF[0]
FLUXCOM_GOSIF=GPP_SIF[1]
MODIS_GOME2=GPP_SIF[2]
MODIS_GOSIF=GPP_SIF[3]
MODIS_TROPO=GPP_SIF[4]


FLUXCOM_GOME2_1 = SetNull(FLUXCOM_GOME2, FLUXCOM_GOME2, "VALUE>30")
maxValueDS = arcpy.GetRasterProperties_management(FLUXCOM_GOME2_1, "MAXIMUM")
maxValue = maxValueDS.getOutput(0)
print "最大值：" + str(maxValue)
minValueDS = arcpy.GetRasterProperties_management(FLUXCOM_GOME2_1, "MINIMUM")
minValue = minValueDS.getOutput(0)
print "最小值：" + str(minValue)
Normal_FLUXCOM_GOME2_1 = (FLUXCOM_GOME2_1 - float(minValue)) / (float(maxValue) - float(minValue))

FLUXCOM_GOSIF_1 = SetNull(FLUXCOM_GOSIF, FLUXCOM_GOSIF, "VALUE>50")
maxValueDS = arcpy.GetRasterProperties_management(FLUXCOM_GOSIF_1, "MAXIMUM")
maxValue = maxValueDS.getOutput(0)
print "最大值：" + str(maxValue)
minValueDS = arcpy.GetRasterProperties_management(FLUXCOM_GOSIF_1, "MINIMUM")
minValue = minValueDS.getOutput(0)
print "最小值：" + str(minValue)
Normal_FLUXCOM_GOSIF_1 = (FLUXCOM_GOSIF_1 - float(minValue)) / (float(maxValue) - float(minValue))

MODIS_GOME2_1 = SetNull(MODIS_GOME2, MODIS_GOME2, "VALUE>30")
maxValueDS = arcpy.GetRasterProperties_management(MODIS_GOME2_1, "MAXIMUM")
maxValue = maxValueDS.getOutput(0)
print "最大值：" + str(maxValue)
minValueDS = arcpy.GetRasterProperties_management(MODIS_GOME2_1, "MINIMUM")
minValue = minValueDS.getOutput(0)
print "最小值：" + str(minValue)
Normal_MODIS_GOME2_1 = (MODIS_GOME2_1 - float(minValue)) / (float(maxValue) - float(minValue))

MODIS_GOSIF_1 = SetNull(MODIS_GOSIF, MODIS_GOSIF, "VALUE>50")
maxValueDS = arcpy.GetRasterProperties_management(MODIS_GOSIF_1, "MAXIMUM")
maxValue = maxValueDS.getOutput(0)
print "最大值：" + str(maxValue)
minValueDS = arcpy.GetRasterProperties_management(MODIS_GOSIF_1, "MINIMUM")
minValue = minValueDS.getOutput(0)
print "最小值：" + str(minValue)
Normal_MODIS_GOSIF_1 = (MODIS_GOSIF_1 - float(minValue)) / (float(maxValue) - float(minValue))

MODIS_TROPO_1 = SetNull(MODIS_TROPO, MODIS_TROPO, "VALUE>50")
maxValueDS = arcpy.GetRasterProperties_management(MODIS_TROPO_1, "MAXIMUM")
maxValue = maxValueDS.getOutput(0)
print "最大值：" + str(maxValue)
minValueDS = arcpy.GetRasterProperties_management(MODIS_TROPO_1, "MINIMUM")
minValue = minValueDS.getOutput(0)
print "最小值：" + str(minValue)
Normal_MODIS_TROPO_1 = (MODIS_TROPO_1 - float(minValue)) / (float(maxValue) - float(minValue))

outRaster = CellStatistics([Normal_FLUXCOM_GOME2_1,Normal_FLUXCOM_GOSIF_1,Normal_MODIS_GOME2_1,Normal_MODIS_GOSIF_1,Normal_MODIS_TROPO_1],"MEAN","")
outRaster.save("C:/Users/lijia/Desktop/1111/new folder/" + "mean.tif")


