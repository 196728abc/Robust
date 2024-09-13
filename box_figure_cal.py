# encoding: utf-8
import arcpy
import numpy as np
from arcpy.sa import *
import os
import glob
import xlwt
import xlrd as xl
# gosif, mW/m^2/sr/nm, 0.5deg, no filled value, have negative value; 2001.1-2020.12
#GO_sif=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\Geosif\GOSIF_0.5deg','*.tif'))

# Gomesif, mW/m^2/sr/nm, 0.5deg, no filled value, have negative value; 2007.1-2018.12
#GOME_sif=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\GOME-2 Monthly 0_5_degree_SIF 2007-2018','*.tif'))

# GOsatsif, mW/m^2/sr/nm, 0.5deg, no filled value, have negative value; 2009.4-2018.12
# GOSAT_sif=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\GOSAT_200904-201812\GOSAT_0.5deg','*.tif'))


# Tropsif, mW/m^2/sr/nm, 0.5deg, no filled value, have negative value; 2018.4-2021.3
# Trop_sif=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\TROPOMI\Tropomi_0.5deg','*.tif'))

# MODIS_GPP, gC m-2 day-1, 0.5deg, no filled value, no negative value; 2001.1-2021-12
# MODIS_GPP=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\MODIS_GPP Monthly 0_5_degree_2001-2021','*.tif'))

# FLUXCOM, gC m-2 day-1, 0.5deg, no filled value, have negative value; 2001.1-2018-12
# FLUXCOM_GPP=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\FLUXCOM\mean','*.tif'))

# VPM_GPP, gC m-2 day-1, 0.5deg, no filled value, no negative value; 2001.1-2016-12
# VPM_GPP=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\VPM_GPP_Monthly_0_5Deg','*.tif'))


arcpy.env.workspace = r'E:\arcpy_workspace'
arcpy.env.overwriteOutput = True

arcpy.CheckOutExtension("Spatial")

#extracted by types and save them


'''
TIF=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\SWR_T_PRE_VPD_SMC\KNN\total_train_test\R2_SIF','*.tif'))

#TIF=TIF[34:49]

landtypes=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\GPP_SIF\GPP_GPP or SIF_SIF\GPP or SIF by gressness\annual NDVI','*.tif'))
landtypes=landtypes[0:9]


for ii in range(0,18):
    R2 = TIF[ii]
    nameT_tif=os.path.basename(R2)
    nameT_tif=nameT_tif.split(".")[0]
   # nameT_tif=nameT_tif.split("_")[1] +'_'+nameT_tif.split("_")[2]+'_'+nameT_tif.split("_")[3]
    #landtypes = landtypess[ii]
    for jj in range(0,9):
        nameT_land=os.path.basename(landtypes[jj])
        nameT_land=nameT_land.split(".")[0]
        nameT_land = nameT_land.split("_")[1]+'_'+nameT_land.split("_")[2]
        outExtractByMask = ExtractByMask(R2, landtypes[jj])
        out_shp = 'E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/meoto/TEST/SWR_T_PRE_VPD_SMC/KNN/total_train_test/R2_SIF/by greenness/'+nameT_tif+'_'+nameT_land+'.shp'
        arcpy.RasterToPoint_conversion(outExtractByMask, out_shp, "VALUE")
        xls = 'E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/meoto/TEST/SWR_T_PRE_VPD_SMC/KNN/total_train_test/R2_SIF/by greenness/'+nameT_tif+'_'+nameT_land+'.xls'
        arcpy.TableToExcel_conversion(out_shp, xls)
'''


#combine all types in xls
'''
xls=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\SWR_T_PRE_VPD_SMC\KNN\total_train_test\R2_GPP\by greenness','*.xls'))

for ii in range(0,18):
    book = xlwt.Workbook()
    sheet = book.add_sheet('sheet1')
    sheet.write(0, 0, '1')    # boreal; croplands
    sheet.write(0, 1, '2')  # dry; DBF
    sheet.write(0, 2, '3')  # temperate; EBF
    sheet.write(0, 3, '4')  # tropical; ENF
    sheet.write(0, 4, '5') #grasslands
    sheet.write(0, 5, '6') #MF
    sheet.write(0, 6, '7') # savannas
    sheet.write(0, 7, '8') #shrublands
    sheet.write(0, 8, '9')  # shrublands
    #sheet.write(0, 9, '10')  # shrublands
    Num=9
    for jj in range(0,Num):
        xls_file=xl.open_workbook(xls[ii*Num+jj])
        xls_sheet=xls_file.sheets()[0]
        col_value=xls_sheet.col_values(2)
        for m in range(len(col_value)):
            sheet.write(m + 1, jj, col_value[m])
    nameT=os.path.basename(xls[ii*Num])
    nameT=nameT.split("_")[0]+"_"+nameT.split("_")[1]+"_"+nameT.split("_")[2]
    book.save(r'E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/meoto/TEST/SWR_T_PRE_VPD_SMC/KNN/total_train_test/R2_GPP/by greenness/111/'+nameT+'.xls')
'''


# 统计R_GPP, R_SIF
xls1=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\SWR_T_PRE_VPD_SMC\KNN\total_train_test\R2_GPP\by greenness\111','*.xls'))
xls2=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\SWR_T_PRE_VPD_SMC\KNN\total_train_test\R2_SIF\by greenness\111','*.xls'))

for ii in range(0,9):
    book = xlwt.Workbook()
    sheet = book.add_sheet('sheet1')
    sheet.write(0, 0, 'R2_MEAN_GPP')  # boreal; croplands
    sheet.write(0, 1, 'R2_STD_GPP')  # dry; DBF
    sheet.write(0, 2, 'R2_MEAN_SIF')  # temperate; EBF
    sheet.write(0, 3, 'R2_STD_SIF')  # tropical; ENF

    xls_file1=xl.open_workbook(xls1[ii])
    xls_file2 = xl.open_workbook(xls2[ii])
    xls_sheet1=xls_file1.sheets()[0]
    xls_sheet2 = xls_file2.sheets()[0]
    for jj in range(9):
        col_value1 = xls_sheet1.col_values(jj)
        col_value2 = xls_sheet2.col_values(jj)
        value1 = col_value1[2:len(col_value1)]
        value2 = col_value2[2:len(col_value2)]
        while '' in value1:
            value1.remove('')
        while '' in value2:
            value2.remove('')
        mean_value1=np.mean(value1)
        std_value1=np.var(value1)
        mean_value2 = np.mean(value2)
        std_value2 = np.var(value2)
        sheet.write(jj + 1, 0, mean_value1)
        sheet.write(jj + 1, 1, std_value1)
        sheet.write(jj + 1, 2, mean_value2)
        sheet.write(jj + 1, 3, std_value2)

    nameT=os.path.basename(xls1[ii])
    #nameT1 = os.path.basename(xls2[ii])
    #nameT=nameT.split("_")[0]+"_"+nameT.split("_")[1]#+"_"+nameT.split("_")[2]
    nameT=nameT.split(".")[0]
    book.save(r'E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/meoto/TEST/SWR_T_PRE_VPD_SMC/KNN/total_train_test/R2_GPP/by greenness/111/Tongji_GPP_SIF_predict/'+nameT+'_Tongji.xls')



'''# 统计ΔR
xls=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\TEST\SWR_T_PRE_VPD_SMC\PLSR\total_train_test\R2_SIF\by greenness\111','*.xls'))
#xls=[xls[0],xls[1],xls[2]]  # FLUXCOM
#xls=[xls[3],xls[4],xls[5]]  # PML
#xls=[xls[6],xls[7],xls[8]]  # VPM
#xls=[xls[9],xls[10],xls[11]]
#xls=[xls[12],xls[13],xls[14]]
xls=[xls[15],xls[16],xls[17]]

book = xlwt.Workbook()
sheet = book.add_sheet('sheet1')
sheet.write(0, 0, 'CSIF')    # boreal; croplands
sheet.write(0, 1, 'STD')  # dry; DBF
sheet.write(0, 2, 'GOMESAT')  # temperate; EBF
sheet.write(0, 3, 'STD')  # tropical; ENF
sheet.write(0, 4, 'TROPOextend') #grasslands
sheet.write(0, 5, 'STD_TROPOextend') #MF


for ii in range(0,3):
    xls_file=xl.open_workbook(xls[ii])
    xls_sheet=xls_file.sheets()[0]
    for jj in range(9):
        col_value = xls_sheet.col_values(jj)
        value=col_value[2:len(col_value)]
        while '' in value:
            value.remove('')
        mean_value=np.mean(value)
        std_value=np.var(value)
        sheet.write(jj + 1, 2*ii, mean_value)
        sheet.write(jj + 1, 2 * ii+1, std_value)
nameT=os.path.basename(xls[ii])
nameT=nameT.split("_")[0]+"_"+nameT.split("_")[1]#+"_"+nameT.split("_")[2]
book.save(r'E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/meoto/TEST/KNN/total_train_test/by greenness/111/TONGJI_GPP_various SIF/'+nameT+'_Tongji.xls')
'''