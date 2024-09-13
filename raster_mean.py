# encoding: utf-8
import arcpy
from arcpy.sa import *
import os
import glob

from xlwt import Workbook

arcpy.env.workspace = r"E:\arcpy_workspace"#影像存放位置
arcpy.CheckOutExtension("Spatial")


FLUXCOM=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\TROPOMI_extend','*.tif'))
#GOME2=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\fesc\MODIS_fesc_NDVI_NIR\NIR','*.tif'))
#NIRv_MOD43=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\fesc/NIRv_MOD43','*.tif'))

#FLUXCOM_1=FLUXCOM[8:len(FLUXCOM):12]
#FLUXCOM_2=FLUXCOM[9:len(FLUXCOM):12]
#FLUXCOM_3=FLUXCOM[10:len(FLUXCOM):12]
#FLUXCOMT=FLUXCOM_1+FLUXCOM_2+FLUXCOM_3
FLUXCOMT=FLUXCOM

'''
landcovers=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\landCover','*.tif'))
landtypes=[landcovers[0],landcovers[1],landcovers[3],landcovers[4],landcovers[5],landcovers[6],landcovers[7],landcovers[8]]
book=Workbook()
sheet=book.add_sheet('MEAN')

for j in range(0,8):
    land=landtypes[j]
    name=os.path.basename(land)
    name=name.split('.')[0]
    sheet.write(0, j, name)
'''
months=range(0,12)
for month in range(0,1):
    outRaster1 = CellStatistics(FLUXCOMT, "MEAN", "")
    #outRaster2 = CellStatistics(GOME2[month:120:12], "MEDIAN", "")
    outRaster1.save("E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/GPP_SIF/GPP_GPP or SIF_SIF/GPP or SIF by gressness/annual SIF/TROPOextend.tif")

'''    
    for j in range(0,8):
        outExtractByMask = ExtractByMask(outRaster1, landtypes[j])
        meanResult = arcpy.GetRasterProperties_management(outExtractByMask, "MEAN")
        meanRes = meanResult.getOutput(0)
        sheet.write(month+1 , j, meanRes)

book.save(r'C:/Users/lijia/Desktop/TEST/relationship bwteen R2 and NDVI/Points_extract by North landcover/NDVI/1_TONGJI.xls')
'''
    #outRaster2.save("C:/Users/lijia/Desktop/mean_GPP_SIF/GOME2_07-16_MEAN/" + "SIF_"+str_month +".tif")




'''
GPP=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\GPP_SIF','*.tif'))
#GPP=GPP[0:3]
1andcover=glob.glob(os.path.join(r'C:\Users\lijia\Desktop\TEST\relationship bwteen R2 and NDVI\total\NDVI','*.tif'))


croplands=landcover[0]
DBF=landcover[1]
# DNF=landcover[2]
EBF=landcover[3]
ENF=landcover[4]
grasslands=landcover[5]
MF=landcover[6]
savannas=landcover[7]
shrublands=landcover[8]
# wetlands=landcover[9]
landtypes=[croplands,DBF,EBF,ENF,grasslands,MF,savannas,shrublands]

landtypes_1=landcover[0:10]
landtypes_2=landcover[10:20]
landtypes_3=landcover[20:30]
landtypes_4=landcover[34:44]
landtypes_5=landcover[45:55]
landtypes_6=landcover[56:66]
landtypes_7=landcover[67:77]
landtypes_8=landcover[78:88]
landtypes_9=landcover[89:99]
landtypes=[landtypes_1,landtypes_2,landtypes_3,landtypes_4,landtypes_5,landtypes_6,landtypes_7,landtypes_8,landtypes_9]
book=Workbook()
sheet=book.add_sheet('MEAN')
sheet1=book.add_sheet('STD')

for j in range(0,9):
    land=landtypes[0][j]
    name=os.path.basename(land)
    name=name.split('_')[3]+'_'+name.split('_')[4][0]
    sheet.write(0, j, name)
    sheet1.write(0, j, name)

for ii in range(0,len(GPP)):
    R2 = GPP[ii]
    landtype=landtypes[ii]
    for jj in range(0,9):
        nameT_land=os.path.basename(landtype[jj])
        nameT_land=nameT_land.split(".")[0]
        outExtractByMask = ExtractByMask(R2, landtype[jj])
        meanValueInfo = arcpy.GetRasterProperties_management(outExtractByMask, 'MEAN')
        meanValue = meanValueInfo.getOutput(0)
        meanValueInfo1 = arcpy.GetRasterProperties_management(outExtractByMask, 'STD')
        meanValue1 = meanValueInfo1.getOutput(0)
        #if meanValue<0:
        #    meanValue=0
        sheet.write(ii+1, jj, meanValue)
        sheet1.write(ii + 1, jj, meanValue1)
book.save(r'C:/Users/lijia/Desktop/TEST/SON/1_TONGJI.xls')
'''

'''
#NDVI=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\fesc_AVHRR/NDVI_AVHRR_2001-2021','*.tif'))
# rasterList5=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\FLUXCOM\mean','*.tif'))
#NIR=glob.glob(os.path.join(rC:/Users\lijia\Desktop\mean_GPP_SIF/NIR_07-15','*.tif'))

R =glob.glob(os.path.join(r'C:\\Users\lijia\Desktop\TEST\PLSR','*.tif'))
#NIR =glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\fesc\fesc_AVHRR/NIR_AVHRR','*.tif'))
#fPAR =glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\fesc\fesc_AVHRR\fPAR_AVHRR','*.tif'))
R=R[54:72]
#n_NDVI=range(0,252)
#n_NIR=range(0,252)
#n_fPAR=range(0,252)

constant=0.08
for i in range(0,18,6):
    for j in range(0,3):
        tif_1=R[i+j]
        tif_2=R[i+j+3]
        nameT=os.path.basename(tif_1)
        name=nameT.split('_')[0]+'_'+nameT.split('_')[1]+'_'+nameT.split('_')[3]
        inRaster_re = Minus(arcpy.Raster(tif_1), arcpy.Raster(tif_2))
        inRaster_re.save("C:/Users/lijia/Desktop/TEST/PLSR/difference of GPP and SIF/" + name)

        #Raster_NDVI=NDVI[n_NDVI[i]]
        #Raster_NIR=NIR[n_NIR[i]]
        #Raster_fPAR = fPAR[n_fPAR[i]]
        #name_T=os.path.basename(Raster_NDVI)
        #inRaster_re = Minus(arcpy.Raster(Raster_NDVI), constant)
        #inRaster_re1 = Times(inRaster_re, arcpy.Raster(Raster_NIR))
        #inRaster_re2 = Divide(inRaster_re1, arcpy.Raster(Raster_fPAR))
        #inRaster_re3 = SetNull(inRaster_re2, inRaster_re2, "VALUE<0")
        #inRaster_re4 = SetNull(inRaster_re3, inRaster_re3, "VALUE>1")

    #inRaster_re4.save("E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/fesc/fesc_AVHRR/fesc_AVHRR_0.08/fesc_" + name_T[5:])
'''
