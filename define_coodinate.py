
import os
import glob
import arcpy
# arcpy.env.workspace=r'E:1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\PAR\0_5deg_direct_PAR'
from arcpy import env
from arcpy.sa import *


'''
prj_N=glob.glob(os.path.join('C:/Users/lijia/Desktop/World_Countries/World_Countries/', "*.prj"))
constant=0.0001
#
Num=['01','02','03','06','07','08','11','12']
outFolder = r'C:/Users/lijia/Desktop/CI/'
outFolder1 = r'C:/Users/lijia/Desktop/CI/0.5/'
for ii in Num:
    inFolder =r'C:\Users\lijia\Desktop\CI\2012'+ii
    arcpy.env.workspace = inFolder
    arcpy.env.overwriteOutput = True
    NCfiles = arcpy.ListRasters('*','tif')
    ppp = "CI_2012_"+ii+".tif"
    arcpy.MosaicToNewRaster_management(NCfiles,outFolder,ppp, "#","16_BIT_SIGNED", "#", "2", "MEAN","FIRST")
'''

''' 
# change coordinate
inFolder =r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\PAR\direct_PAR'
arcpy.env.workspace = inFolder
arcpy.env.overwriteOutput = True

NCfiles = arcpy.ListRasters('*','tif')
outFolder = 'E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/PAR/direct_PAR/111/'
constant=0.0001
prj_N=glob.glob(os.path.join('C:/Users/lijia/Desktop/World_Countries/World_Countries/', "*.prj"))
for ras in NCfiles:
    print str(ras)
    ppp = os.path.basename(ras)
    out_raster1 = outFolder+ras

    arcpy.ProjectRaster_management(ras, out_raster1, prj_N[0])
print('OK')
'''


# define coordinate
inFolder =r'C:\Users\lijia\Desktop\PAR\scatter PAR'
#inFolder =r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\fesc\MODIS_fesc_NDVI_NIR\NIR'
arcpy.env.workspace = inFolder
arcpy.env.overwriteOutput = True

NCfiles = arcpy.ListRasters('*','tif')
outFolder = 'C:/Users/lijia/Desktop/PAR/scatter PAR/111/'
prj_N=glob.glob(os.path.join('C:/Users/lijia/Desktop/World_Countries/World_Countries/', "*.shp"))


for ras in NCfiles:
    print str(ras)
    #ppp = os.path.basename(ras)
    #out_raster1 = outFolder+ras
    #img_SR = arcpy.Describe(ras).spatialReference.name
    arcpy.DefineProjection_management(ras, prj_N[0])
    nameT=os.path.basename(ras)
    out_raster=outFolder+nameT
    arcpy.Shift_management(ras, out_raster,"-180", "90")

print('OK')


'''
tmp=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\CRU\tmp','*.tif'))
vap=glob.glob(os.path.join(r'E:\1_canopy_photosynthesis\photosynthesis_and_Global_Climate\SIF\data\meoto\CRU\vap','*.tif'))

arcpy.env.workspace = 'E:/arcpy_workspace/'
arcpy.env.overwriteOutput = True

constant1 = 17.67
constant2 = 243.5
constant3 = 6.112
OutRasters="E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/meoto/CRU/VPD/"
for i in range(0,252):
    name_T=os.path.basename(tmp[i])
    Out=OutRasters+"VPD_"+name_T[4:]
          
    Le = Times(arcpy.Raster(tmp[i]),constant1)
    Le1 = Plus(arcpy.Raster(tmp[i]), constant2)
    Le2 = Divide(Le,Le1)
    Le3 = Exp(Le2)
    Es = Times(Le3,constant3)
    VPD = Minus(Es,arcpy.Raster(vap[i]))
    VPD.save(Out)
'''