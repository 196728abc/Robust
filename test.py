# -*- coding: UTF-8 -*-

'''
import glob
import numpy as np
import pandas as pd
import os
import xlwt
from collections import Counter
from sklearn import preprocessing
from pandas.core.frame import DataFrame
'''


import os
import glob
import arcpy
from arcpy.sa import *

arcpy.CheckOutExtension('Spatial')
arcpy.env.workspace = r'E:\arcpy_workspace'


GPP = glob.glob(os.path.join(r"E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\FLUXCOM","*.tif"))
SIF = glob.glob(os.path.join(r"E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\GOSAT_200904-201812\true data","*.tif"))
PAR = glob.glob(os.path.join(r"E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\SWR", "*.tif"))
TMP = glob.glob(os.path.join(r"E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\TMP", "*.tif"))
PRE = glob.glob(os.path.join(r"E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\PRE", "*.tif"))
VPD = glob.glob(os.path.join(r"E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\VPD", "*.tif"))
SM = glob.glob(os.path.join(r"E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\SM", "*.tif"))



landcoverss=glob.glob(os.path.join(r"E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\GPP_SIF\GPP_GPP or SIF_SIF\GPP or SIF by gressness\annual NDVI", "*.tif"))
#landcovers=[landcoverss[0],landcoverss[1],landcoverss[3],landcoverss[4],landcoverss[5],landcoverss[6],landcoverss[7],landcoverss[8]]
# shp=glob.glob(os.path.join(r'E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/meoto/GLDAS_NOAH/Point_comprehensive_corr/temp_shp/111/','*.shp'))
landcovers=landcoverss[0:9]
name_landcover=['0_NDVI_0_1','0_NDVI_0_2','0_NDVI_0_3','0_NDVI_0_4','0_NDVI_0_5','0_NDVI_0_6','0_NDVI_0_7','0_NDVI_0_8','0_NDVI_0_9']


KK=1

arcpy.CheckOutExtension("ImageAnalyst")  # 检查许可

for num_land in range(0, 9):
    #cover_name=os.path.basename(landcovers[num_land])
    #cover_name = cover_name.split('.')[0]
    #nameTT = 'GOME2'
    #nameT = nameTT +'.shp'
    Points_shp = "E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/meoto/TEST/Points/"+name_landcover[num_land]+'/GOSAT.shp'
    ras = GPP[8]
    ras =  ExtractByMask(ras, landcovers[num_land])
    #outname = os.path.join(Points_shp, nameT)  # 合并输出文件名+输出路径
    out_point1 = arcpy.RasterToPoint_conversion(ras, Points_shp, "VALUE")
    # 24:192
    num_1 = range(0,63)
    lists_1 = []
    str_list = []
    for i in num_1:
        flux_gpp = SIF[i]
        outExtractByMask = ExtractByMask(flux_gpp, landcovers[num_land])
        str = os.path.basename(flux_gpp)
        # str = str.rsplit('.', 1)[0]
        str = 'SF_' + str[4:11]
        # str='M_'+str
        str_list.append(str)
        AA = [outExtractByMask, str]
        lists_1.append(AA)

    if KK==1:
        # 0:168
        num_2 = range(99,162)
        lists_2 = []
        # str_list_2=[]
        for i in num_2:
            gosif = PAR[i]
            outExtractByMask1 = ExtractByMask(gosif, landcovers[num_land])
            str = os.path.basename(gosif)
            str = 'PA_' + str[4:11]
            # str = str.rsplit('.', 1)[0]
            str_list.append(str)
            AA = [outExtractByMask1, str]
            lists_2.append(AA)

        num_3 = range(99,162)
        lists_3 = []
        # str_list_2=[]
        for i in num_3:
            gosif = TMP[i]
            outExtractByMask1 = ExtractByMask(gosif, landcovers[num_land])
            str = os.path.basename(gosif)
            str = 'TM_' + str[4:11]
            # str = str.rsplit('.', 1)[0]
            str_list.append(str)
            AA = [outExtractByMask1, str]
            lists_3.append(AA)

        num_4 = range(99,162)
        lists_4 = []
        # str_list_2=[]
        for i in num_4:
            gosif = PRE[i]
            outExtractByMask1 = ExtractByMask(gosif, landcovers[num_land])
            str = os.path.basename(gosif)
            str = 'PR_' + str[4:11]
            # str = str.rsplit('.', 1)[0]
            str_list.append(str)
            AA = [outExtractByMask1, str]
            lists_4.append(AA)

        num_5 = range(99,162)
        lists_5 = []
        # str_list_2=[]
        for i in num_5:
            gosif = VPD[i]
            outExtractByMask1 = ExtractByMask(gosif, landcovers[num_land])
            str = os.path.basename(gosif)
            str = 'VD_' + str[4:11]
            # str = str.rsplit('.', 1)[0]
            str_list.append(str)
            AA = [outExtractByMask1, str]
            lists_5.append(AA)

        num_6 = range(99,162)
        lists_6 = []
        # str_list_2=[]
        for i in num_6:
            gosif = SM[i]
            outExtractByMask1 = ExtractByMask(gosif, landcovers[num_land])
            str = os.path.basename(gosif)
            str = 'SM_' + str[3:10]
            # str = str.rsplit('.', 1)[0]
            str_list.append(str)
            AA = [outExtractByMask1, str]
            lists_6.append(AA)

    else:
        # 0:168
        num_2 = range(24,204)
        lists_2 = []
        # str_list_2=[]
        for i in num_2:
            gosif = PRE[i]
            outExtractByMask1 = ExtractByMask(gosif, landcovers[num_land])
            str = os.path.basename(gosif)
            str = 'PR_' + str[4:11]
            # str = str.rsplit('.', 1)[0]
            str_list.append(str)
            AA = [outExtractByMask1, str]
            lists_2.append(AA)

        num_3 = range(24,204)
        lists_3 = []
        # str_list_2=[]
        for i in num_3:
            gosif = VPD[i]
            outExtractByMask1 = ExtractByMask(gosif, landcovers[num_land])
            str = os.path.basename(gosif)
            str = 'VD_' + str[4:11]
            # str = str.rsplit('.', 1)[0]
            str_list.append(str)
            AA = [outExtractByMask1, str]
            lists_3.append(AA)

        num_4 = range(24,204)
        lists_4 = []
        # str_list_2=[]
        for i in num_4:
            gosif = SM[i]
            outExtractByMask1 = ExtractByMask(gosif, landcovers[num_land])
            str = os.path.basename(gosif)
            str = 'SM_' + str[3:10]
            # str = str.rsplit('.', 1)[0]
            str_list.append(str)
            AA = [outExtractByMask1, str]
            lists_4.append(AA)


    ExtractMultiValuesToPoints(out_point1, lists_1, "NONE")
    ExtractMultiValuesToPoints(out_point1, lists_2, "NONE")
    ExtractMultiValuesToPoints(out_point1, lists_3, "NONE")
    ExtractMultiValuesToPoints(out_point1, lists_4, "NONE")
    ExtractMultiValuesToPoints(out_point1, lists_5, "NONE")
    ExtractMultiValuesToPoints(out_point1, lists_6, "NONE")

    #xls = 'C:/Users/lijia/Desktop/TEST/Point_FLUXCOM_GOSIF/'+name_landcover[num_land]+'/'+nameTT+'.xls'
    #arcpy.TableToExcel_conversion(out_point1, xls)

print()

