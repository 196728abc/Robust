
import os
import arcpy

hdf_file_path=r"E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/revised_LUE_GPP"
output_file="E:/1_canopy_photosynthesis/photosynthesis_and_Global_Climate/SIF/data/revised_LUE_GPP/111/"
#output_file1="C:/Users/lijia/Desktop/PAR/scatter PAR/"

#hdf_file_name_list=os.listdir(hdf_file_path)
NCfiles = [nc_file for nc_file in os.listdir(hdf_file_path) if nc_file.endswith(".hdf")]
#hh=hdf_file_name_list[0]
#fff=hh[52:58]
#fff=hh[47:53]
Num=range(0,14)
for i in range(0,1):
    hdf_file=NCfiles[i]

    Year_str=hdf_file[16:20]
    month_str=hdf_file[20:23]
    '''
    month=(i+1)%12
    if month==0:
        month=12
    if month<10:
        month_str='0'+str(month)
    else:
        month_str=str(month)
    '''
    tif_file_name1 = 'GPP_'+Year_str+month_str+".tif"
    #tif_file_name2 = 'end_'+Year_str+".tif"
    data1=arcpy.ExtractSubDataset_management(hdf_file_path+'/'+hdf_file,output_file+tif_file_name1,"0")
    #data2=arcpy.ExtractSubDataset_management(hdf_file_path+'/'+hdf_file,output_file+tif_file_name2,"1")


'''
for hdf_file in hdf_file_name_list:
    if os.path.isdir(hdf_file_path+hdf_file):
        file_name_temp=hdf_file
        hdf_file_name_list_new=os.listdir(hdf_file_path+hdf_file)
        for hdf_file in hdf_file_name_list_new:
            tif_file_name=hdf_file[52:58]+".tif"
          #  data=arcpy.ExtractSubDataset_management(hdf_file_path+file_name_temp+'/'+hdf_file,tif_file_path+tif_file_name,"0;4")
    else:
            tif_file_name=hdf_file[52:58]+".tif"
        #    data=arcpy.ExtractSubDataset_management(hdf_file_path+hdf_file,tif_file_path+tif_file_name,"0;4")
'''