#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 功能：将nc文件（in_nc）中的目标时间维度（times）中的variable变量导出为以"原文件名+时间维度索引"为新文件名的栅格文件
# 提示：导入脚本前强烈建议将本代码中的所有中文字符删除！！！
import os
import glob
import time
import arcpy
from arcpy.sa import *
from datetime import datetime, timedelta


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlwt
# from statsmodels.formula.api import ols

GPP_tot=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\GPP_SIF\Points extract by landcover\VPM_TROPOextend','*.shp'))

arcpy.CheckOutExtension("Spatial")
for i in range(0,16):

    nameT=os.path.basename(GPP_tot[i])
    nameT=nameT.split('.')[0]
    xls = 'E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/GPP_SIF/Points extract by landcover/VPM_TROPOextend/'+nameT+'.xls'
    arcpy.TableToExcel_conversion(GPP_tot[i], xls)
    print()


'''
    dry_1 = dry[i]
    str = os.path.basename(dry_1)
    rectExtract1 = ExtractByRectangle(dry_1, inRectangle, "INSIDE")
    rectExtract1.save("C:/Users/lijia/Desktop/gpp_data/dry_1/" + str)

    temperate_1 = temperate[i]
    str = os.path.basename(temperate_1)
    rectExtract1 = ExtractByRectangle(temperate_1, inRectangle, "INSIDE")
    rectExtract1.save("C:/Users/lijia/Desktop/gpp_data/temperate_1/" + str)

    tropical = tropical[i]
    str = os.path.basename(tropical)
    rectExtract1 = ExtractByRectangle(tropical, inRectangle, "INSIDE")
    rectExtract1.save("C:/Users/lijia/Desktop/gpp_data/tropical_1/" + str)
'''
'''
climate_landcover=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\climate_landcover','*.tif'))

boreal_croplands = climate_landcover[0]
boreal_DBF = climate_landcover[1]
boreal_DNF = climate_landcover[2]
boreal_EBF = climate_landcover[3]
boreal_ENF = climate_landcover[4]
boreal_grasslands = climate_landcover[5]
boreal_MF=climate_landcover[6]
boreal_savannas=climate_landcover[7]
boreal_shrublands=climate_landcover[8]
boreal_wetlands=climate_landcover[9]

dry_croplands = climate_landcover[10]
dry_DBF = climate_landcover[11]
dry_DNF = climate_landcover[12]
dry_EBF = climate_landcover[13]
dry_ENF = climate_landcover[14]
dry_grasslands = climate_landcover[15]
dry_MF=climate_landcover[16]
dry_savannas=climate_landcover[17]
dry_shrublands=climate_landcover[18]
dry_wetlands=climate_landcover[19]

temperate_croplands = climate_landcover[20]
temperate_DBF = climate_landcover[21]
temperate_DNF = climate_landcover[22]
temperate_EBF = climate_landcover[23]
temperate_ENF = climate_landcover[24]
temperate_grasslands = climate_landcover[25]
temperate_MF=climate_landcover[26]
temperate_savannas=climate_landcover[27]
temperate_shrublands=climate_landcover[28]
temperate_wetlands=climate_landcover[29]

tropical_croplands = climate_landcover[30]
tropical_DBF = climate_landcover[31]
tropical_DNF = climate_landcover[32]
tropical_EBF = climate_landcover[33]
tropical_ENF = climate_landcover[34]
tropical_grasslands = climate_landcover[35]
tropical_MF=climate_landcover[36]
tropical_savannas=climate_landcover[37]
tropical_shrublands=climate_landcover[38]
tropical_wetlands=climate_landcover[39]


# FLUXCOM_GOME2； 2011.1-2018.12
MODIS_GPP_GO_SIF=[]
indexes_SIF=range(48,144)  #total months
indexes_GPP=range(120,216)
i_total=range(0,96)
months=range(0,12)
namecode=[1,2,3,4,5,6,7,8,9,10,11,12]
for i in i_total:
    index1 = indexes_SIF[i]
    raster_GPP=FLUXCOM_GPP[index1]
    outRaster=ExtractByMask(raster_GPP, boreal_croplands)

    index2 = indexes_GPP[i]

    sif_11 = SetNull(GOME_sif[index1], GOME_sif[index1], "VALUE<0")
    sif_1 = SetNull(sif_11, sif_11, "VALUE=0")
    gpp_11 = SetNull(FLUXCOM_GPP[index2], FLUXCOM_GPP[index2], "VALUE<0")
    gpp_1 = SetNull(gpp_11, gpp_11, "VALUE=0")
    aa=Divide(gpp_1,sif_1)
    MODIS_GPP_GO_SIF.append(aa)
#MODIS_GPP_GO_SIF_mean = []
MODIS_GPP_GO_SIF = np.array(MODIS_GPP_GO_SIF)
for month in months:
    namecode1=namecode[month]
    if namecode1<10:
        namecode_str='0'+str(namecode1)
    else:
        namecode_str=str(namecode1)
    outRaster = CellStatistics(MODIS_GPP_GO_SIF[range(month,96,12)], "MEAN", "")
    #MODIS_GPP_GO_SIF_mean.append(outRaster)
    outRaster.save("C:/Users/lijia/Desktop/gpp_data/" + "gpp_sif_" + namecode_str + ".tif")
print()

book=xlwt.Workbook()
sheet=book.add_sheet('sheet1')

for m in range(len(list1)):
    sheet.write(m,0,list1[m])
    sheet.write(m,1,list2[m])
    sheet.write(m,2,list3[m])
'''
