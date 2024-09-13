# encoding: utf-8
import arcpy
from arcpy.sa import *
import os
import glob



#GPP=glob.glob(os.path.join(r'C:\Users\lijia\Desktop\TEST\SON\NDVI','*.tif'))
GPP=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\PDSI\111','*.tif'))

output_path1 = "E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/PDSI/111/111/"

arcpy.env.workspace = r'E:\arcpy_workspace'
arcpy.env.overwriteOutput = True

#shp=glob.glob(os.path.join(r'C:\Users\lijia\Desktop\World_Countries\World_Countries','*.shp'))

arcpy.CheckOutExtension("Spatial")

for raster in GPP:
    nameT = os.path.basename(raster)

    #str_year = nameT.split('_')[1]
    #str_month=nameT.split('_')[2]+'_'+nameT.split('_')[3]
    '''
    str_month = str_month.split('.')[0]
    if int(str_month)<10:
        aa='0'+str(str_month)
    else:
        aa=str(str_month)
         '''
    #nameT='LAI_'+str_year+'_'+str_month

    #raster1 = SetNull(raster,raster,'VALUE<0')
    #raster2 = SetNull(raster1, raster1, 'VALUE>365')
    #inRectangle = Extent(-180, -60, 180, 90)
    #raster2=ExtractByRectangle(raster, inRectangle,"INSIDE")
    #outExtractByMask = ExtractByMask(raster2, shp[0])
    #raster2.save(output_path1 + nameT)
    arcpy.Resample_management(raster, output_path1+nameT, str(0.5), "bilinear")#重采样后的分辨率以及重采样的方法

'''
list=[]
for i in range(16):
    list.append(NDVI_T[i*92:92*i+92])
for i in range(0,len(list)):
    SIF=list[i]
    if i<9:
        str_year="200"+str(i+1)
    else:
        str_year = "20" + str(i + 1)
    #day=[8,7,8,7,8,7,8,8,8,7,8,8]
    day=[0,8,15,23,30,38,45,53,61,69,76,84,92]

    list1=[]
    for j in range(12):
        if j<9:
            str_month='0'+str(j+1)
        else:
            str_month=str(j+1)
        list1 = SIF[day[j]:day[j+1]]
        outRaster = CellStatistics(list1, "MEAN", "")
        outRaster.save("C:/Users/lijia/Desktop/6387494/111/" + "SIF_" + str_year+'_'+str_month + ".tif")
print()
'''

'''
for i in range(0,18):
    #list1=[]
    #for j in range(len(GPP_SIF)):
    TT=GPP[i]
    nameT_1=os.path.basename(TT)
    nameT_1=nameT_1.split('.')[0]
    #outRaster = ExtractByMask(arcpy.Raster(TT), landtypes[i])
    #nameT_1 = nameT_1.split('_')[1] + '_'+nameT_1.split('_')[2]+'_'+landtypes_names[i]
    for j in range(0,9):
        #str_year = nameT.split('_')[1]
        #str_month=nameT.split('_')[2]
        #str_month = str_month.split('.')[0]
            #if i+1==int(str_month):
             #   list1.append(TT)
        #if int(str_month)<10:
        #    aa='0'+str(str_month)
        #else:
        #    aa=str(str_month)
        #nameT=nameT_1+'_'+landtypes_names[j]+'.shp'
        #nameT_shp = nameT_1 + '_' + landtypes_names[j] + '.shp'
        #out = output_path1 + nameT
        out_shp=output_path1+nameT_1+'_'+NDVIypes_names[j]+'.shp'
        xls_name=output_path1+nameT_1+'_'+NDVIypes_names[j]+'.xls'
        #outRaster=Divide(arcpy.Raster(TT),arcpy.Raster(TT1))
        try:
            outRaster1 = ExtractByMask(arcpy.Raster(TT), NDVItypes[j])
            arcpy.RasterToPoint_conversion(outRaster1, out_shp, "VALUE")
            #arcpy.Clip_analysis(TT, landtypes[j], out_shp)
            arcpy.TableToExcel_conversion(out_shp, xls_name)
            #outRaster = ExtractByMask(arcpy.Raster(TT), landtypes[j])
            #arcpy.RasterToPoint_conversion(outRaster, out_shp, "VALUE")
            #outRaster.save(out)
        except:
            print("***********")
        #outRaster = CellStatistics(list1, "MEAN", "")
        #arcpy.Resample_management(arcpy.Raster(TT), out, str(0.5), "CUBIC")#重采样后的分辨率以及重采样的方法
        print()
'''
#NDVI11=glob.glob(os.path.join(r'C:\Users\lijia\Desktop\TEST\relationship bwteen R2 and NDVI\Points_extract by North landcover\NDVI','*.tif'))
'''
#inMaskData=NDVI11[0]
#list=['FLUXCOM_CSIF','FLUXCOM_GOMESAT','FLUXCOM_TROPOextend','PML_CSIF','PML_GOMESAT','PML_TROPOextend','VPM_CSIF','VPM_GOMESAT','VPM_TROPOextend']
#constant=0.0001
for i in range(0,3):
    for j in range(0,3):
        GPP_SIF_1=GPP_SIF[i]
        nameT1=os.path.basename(GPP_SIF_1)
        GPP_INDEX_1=GPP_INDEX[j]
        nameT2 = os.path.basename(GPP_INDEX_1)
        nameT=nameT1.split('.')[0][3:]+'_'+nameT2.split('_')[1]+'.tif'
        outraster=Minus(arcpy.Raster(GPP_SIF_1),arcpy.Raster(GPP_INDEX_1))
        outraster.save(output_path1+nameT)

    # NIR_1=fPAR_T[i]
   # if i % 12 == 0:
   #     year=year+1
   #     month = (i % 12)+1
   # else:
   #     year = year
   #     month = (i % 12) + 1
    #nameT = os.path.basename(NDVI_1)
    #aa=arcpy.Raster(NDVI_1)
    #outExtractByMask = ExtractByMask(aa, inMaskData)
    #outExtractByMask.save(output_path1+nameT)

    #nameT='NIRv'+nameT[4:]
    #str_year = nameT.split("-")[1]
    #str_month = nameT.split(".")[0][9:]
    #nameT='SIF_'+str_year+'_'+str_month+'.tif'
    #str_year=nameT.split(".")[0][4:8]
    #str_month = nameT.split(".")[0][9:]
    #if int(str_month)<10:
    #    str_month= "0"+str_month
    #nameT='VPD_'+str_year+'_'+str_month+'.tif'

   # inRaster = arcpy.Raster(NDVI_1)
   # outCon = Con(IsNull(inRaster), 0, inRaster)
   # outCon.save("C:\\Users\lijia\Desktop\AAA.tif")

   # NIRv_1 = NIRv_T[NUM_NIRv[i]]
   # NDVI_2 = NDVI[i+1]
   # NIR = NIR_T[num_NIR[i]]
   # fPAR = fPAR_T[num_fPAR[i]]

  #  SIF=arcpy.Raster(SIF_1)
    #if int(str_month)<10:
     #   str_month = "0"+str_month
  #  name = nameT[5:]
  #  outRaster = CellStatistics([NDVI_1,NDVI_2], "MEAN", "")
  #  inRaster_re=Minus(arcpy.Raster(NDVI_1),0.08)
  #  inRaster_re1 = Plus(arcpy.Raster(NDVI_1), arcpy.Raster(NIR_1))

    #inRaster_re1 = Times(arcpy.Raster(NDVI_1), arcpy.Raster(NIR_1))
    inRaster_re1 = SetNull(arcpy.Raster(FLUXCOM[i]), arcpy.Raster(FLUXCOM[i]), "VALUE<0")
    inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE>=0.1")
    inRaster_re2.save(output_path1 + 'NDVI_'+list[i]+'_0_1.tif')
    inRaster_re1 = SetNull(arcpy.Raster(FLUXCOM[i]), arcpy.Raster(FLUXCOM[i]), "VALUE<0.1")
    inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE>=0.2")
    inRaster_re2.save(output_path1 + 'NDVI_'+list[i]+'_0_2.tif')
    inRaster_re1 = SetNull(arcpy.Raster(FLUXCOM[i]), arcpy.Raster(FLUXCOM[i]), "VALUE<0.2")
    inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE>=0.3")
    inRaster_re2.save(output_path1 + 'NDVI_'+list[i]+'_0_3.tif')
    inRaster_re1 = SetNull(arcpy.Raster(FLUXCOM[i]), arcpy.Raster(FLUXCOM[i]), "VALUE<0.3")
    inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE>=0.4")
    inRaster_re2.save(output_path1 + 'NDVI_'+list[i]+'_0_4.tif')
    inRaster_re1 = SetNull(arcpy.Raster(FLUXCOM[i]), arcpy.Raster(FLUXCOM[i]), "VALUE<0.4")
    inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE>=0.5")
    inRaster_re2.save(output_path1 + 'NDVI_'+list[i]+'_0_5.tif')
    inRaster_re1 = SetNull(arcpy.Raster(FLUXCOM[i]), arcpy.Raster(FLUXCOM[i]), "VALUE<0.5")
    inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE>=0.6")
    inRaster_re2.save(output_path1 + 'NDVI_'+list[i]+'_0_6.tif')
    inRaster_re1 = SetNull(arcpy.Raster(FLUXCOM[i]), arcpy.Raster(FLUXCOM[i]), "VALUE<0.6")
    inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE>=0.7")
    inRaster_re2.save(output_path1 + 'NDVI_'+list[i]+'_0_7.tif')
    inRaster_re1 = SetNull(arcpy.Raster(FLUXCOM[i]), arcpy.Raster(FLUXCOM[i]), "VALUE<0.7")
    inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE>=0.8")
    inRaster_re2.save(output_path1 + 'NDVI_'+list[i]+'_0_8.tif')
    inRaster_re1 = SetNull(arcpy.Raster(FLUXCOM[i]), arcpy.Raster(FLUXCOM[i]), "VALUE<0.8")
    inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE>=0.9")
    inRaster_re2.save(output_path1 + 'NDVI_'+list[i]+'_0_9.tif')
    '''

    #inRaster_re3 = SetNull(inRaster_re2, inRaster_re2, "VALUE>1")
    #NDVI_11 = ExtractByMask(inRaster_re3, shp[0])
   # inRaster_re1 = Minus(NDVI_11,constant1)

    #inRaster_re2 = SetNull(inRaster_re1, inRaster_re1, "VALUE<0")


    #arcpy.Resample_management(NDVI_1, output_path1+nameT, str(0.5), "CUBIC")#重采样后的分辨率以及重采样的方法

  #  arcpy.Resample_management(NDVI_1, out, str(0.5), "CUBIC")#重采样后的分辨率以及重采样的方法
   # arcpy.Resample_management(inRaster_re3, out, str(0.5), "CUBIC")#重采样后的分辨率以及重采样的方法




'''
aa_1=raster.split(".")
    year_num=int(aa_1[2])
    month_num=int(aa_1[3][1:])
    if year_num % 4 == 0:
        er_yue=29
    else:
        er_yue = 28
    if month_num == 2:
        constant=er_yue
    elif month_num in aaa:
        constant=31
    else:
        constant=30
'''


'''
MODIS_GPP=glob.glob(os.path.join(r'E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/TROPOMI/Tropomi_0.5deg','*.tif'))
world_shp=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\climate types\type of Climate','*.tif'))

output_path_boreal="E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/TROPOMI/Tropomi_0.5deg/boreal/"
output_path_dry="E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/TROPOMI/Tropomi_0.5deg/dry/"
output_path_temperate="E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/TROPOMI/Tropomi_0.5deg/temperate/"
output_path_tropical="E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/TROPOMI/Tropomi_0.5deg/tropical/"



boreal=world_shp[0]
dry=world_shp[1]
temperate=world_shp[2]
tropical=world_shp[3]

for raster in MODIS_GPP:
    nameT = os.path.basename(raster)
    outExtractByMask = ExtractByMask(raster, boreal)
    outExtractByMask.save(output_path_boreal + nameT)

    outExtractByMask = ExtractByMask(raster, dry)
    outExtractByMask.save(output_path_dry + nameT)

    outExtractByMask = ExtractByMask(raster, temperate)
    outExtractByMask.save(output_path_temperate + nameT)

    outExtractByMask = ExtractByMask(raster, tropical)
    outExtractByMask.save(output_path_tropical + nameT)
   # out = output_path1 + nameT
   # arcpy.Resample_management(outExtractByMask, out, str(0.5), "CUBIC")#重采样后的分辨率以及重采样的方法
   # outExtractByMask.save("E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/climate_landcover/" + Climate_str+"_"+landCover_str+ ".tif")

print()

'''


'''
fesc=glob.glob(os.path.join(r'C:/Users\lijia\Desktop/NIR-20220531T090043Z-001\fesc','*.tif'))
landcovers=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\landCover','*.tif'))

output_path_croplands="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/croplands/"
output_path_DBF="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/DBF/"
output_path_DNF="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/DNF/"
output_path_EBF="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/EBF/"
output_path_ENF="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/ENF/"
output_path_grasslands="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/grasslands/"
output_path_MF="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/MF/"
output_path_savannas="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/savannas/"
output_path_shrublands="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/shrublands/"
output_path_wetlands="C:/Users/lijia/Desktop/NIR-20220531T090043Z-001/fesc/wetlands/"

croplands=landcovers[0]
DBF=landcovers[1]
DNF=landcovers[2]
EBF=landcovers[3]
ENF=landcovers[4]
grasslands=landcovers[5]
MF=landcovers[6]
savannas=landcovers[7]
shrublands=landcovers[8]
wetlands=landcovers[9]

for raster in fesc:
    nameT = os.path.basename(raster)
    outExtractByMask = ExtractByMask(raster, croplands)
    outExtractByMask.save(output_path_croplands + nameT)

    outExtractByMask = ExtractByMask(raster, DBF)
    outExtractByMask.save(output_path_DBF + nameT)

    outExtractByMask = ExtractByMask(raster, DNF)
    outExtractByMask.save(output_path_DNF + nameT)

    outExtractByMask = ExtractByMask(raster, EBF)
    outExtractByMask.save(output_path_EBF + nameT)

    outExtractByMask = ExtractByMask(raster, ENF)
    outExtractByMask.save(output_path_ENF + nameT)

    outExtractByMask = ExtractByMask(raster, grasslands)
    outExtractByMask.save(output_path_grasslands + nameT)

    outExtractByMask = ExtractByMask(raster, MF)
    outExtractByMask.save(output_path_MF + nameT)

    outExtractByMask = ExtractByMask(raster, savannas)
    outExtractByMask.save(output_path_savannas + nameT)

    outExtractByMask = ExtractByMask(raster, shrublands)
    outExtractByMask.save(output_path_shrublands + nameT)

    outExtractByMask = ExtractByMask(raster, wetlands)
    outExtractByMask.save(output_path_wetlands + nameT)

   # out = output_path1 + nameT
   # arcpy.Resample_management(outExtractByMask, out, str(0.5), "CUBIC")#重采样后的分辨率以及重采样的方法
   # outExtractByMask.save("E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/climate_landcover/" + Climate_str+"_"+landCover_str+ ".tif")

print()
'''
