## -- Code for compliation of data for analysis and export of PACs compatible dicoms  -- ##
import numpy as np
import os
import pydicom
import FlipCal
import Vent_Analysis
from helpers.vent_combine import tile_and_save_rgb_dicom_2x8
from helpers.GX_to_DICOM import *
from helpers.pdf_to_dicom import *


## -- Populate with the relevant paths -- ##
parent_dir = "c:/tmp"
xenon_pre_folder = "C:/tmp/xenon pre/DICOM/EXP00000/"
xenon_pos_folder = "C:/tmp/xenon post/DICOM/EXP00000"
proton_pre_folder = "C:/tmp/proton exp/DICOM/EXP00000"
proton_pos_folder = proton_pre_folder
dicom_pre_template_path = os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0])
dicom_pos_template_path = os.path.join(xenon_pos_folder,os.listdir(xenon_pre_folder)[0])
calibration_twix_path = "C:/tmp/meas_MID00217_FID16268_XE_fid_calibration_dyn.dat"
xenoview_pre_folder = "C:/tmp/Xenoview_pre"
xenoview_pos_folder = "C:/tmp/Xenoview_post"
xenoview_pre_pdf = "C:/tmp/Xenoview_pre.pdf"
xenoview_pos_pdf = "C:/tmp/Xenoview_post.pdf"
GX_folder = "C:/tmp/GX"
GX_pdf = "C:/tmp/GX/Dangelmaier, Anja Elizabeth _report.pdf"
xenoview_pre_VDP = 27.6
xenoview_pos_VDP = 26.9


pre_mask_folder = os.path.join(xenoview_pre_folder, 'lung-segmentation-legacy/lung-segmentation-volume')
pos_mask_folder = os.path.join(xenoview_pos_folder, 'lung-segmentation-legacy/lung-segmentation-volume')

## -- Create the PACs upload folder -- ##
PACS_dir = os.path.join(parent_dir,'PACs')
os.makedirs(PACS_dir,exist_ok=True)


## -- Analyze the FlipCal -- ##
FA = FlipCal.FlipCal(twix_path = calibration_twix_path)
FA.process()
FA.completeExport(parent_dir=parent_dir,dummy_dicom_path=os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0]))
FA.dicomPrintout(dummy_dicom_path = dicom_pre_template_path,save_path = os.path.join(PACS_dir,'FlipCal'))


## -- Do the in-house Vent Analysis -- ##
VentPre = Vent_Analysis.Vent_Analysis(xenon_path=xenon_pre_folder,mask_path=pre_mask_folder,proton_path=proton_pre_folder)
VentPre.calculate_VDP()
VentPre.completeExport(f"{parent_dir}/Vent_pre",dicom_template_path=dicom_pre_template_path,fileName=None,SlicLocs=None)

VentPos = Vent_Analysis.Vent_Analysis(xenon_path=xenon_pos_folder,mask_path=pos_mask_folder,proton_path=proton_pos_folder)
VentPos.calculate_VDP()
VentPos.completeExport(f"{parent_dir}/Vent_pos",dicom_template_path=dicom_pre_template_path,fileName=None,SlicLocs=None)


## -- Tile the Vents -- ##
dirs = (xenon_pre_folder, 
        xenon_pos_folder, 
        f"{parent_dir}/Vent_pre/VentDicoms", 
        f"{parent_dir}/Vent_pos/VentDicoms", 
        os.path.join(xenoview_pre_folder,'anatomical-ventilation-map/anatomical-ventilation-map-volume'), 
        os.path.join(xenoview_pos_folder,'anatomical-ventilation-map/anatomical-ventilation-map-volume'))
stats = ('XXXXXXXXX',
    #VentPre.metadata['PatientName'],
    VentPre.metadata['StudyDate'],
    VentPre.metadata['LungVolume'],
    np.round(VentPre.metadata['LungVolume']*VentPre.metadata['VDP']/100*1000,1),
    np.round(VentPre.metadata['VDP']),
    xenoview_pre_VDP,
    VentPos.metadata['LungVolume'],
    np.round(VentPos.metadata['LungVolume']*VentPos.metadata['VDP']/100*1000,1),
    np.round(VentPos.metadata['VDP']),
    xenoview_pos_VDP,
)

tile_and_save_rgb_dicom_2x8(dirs, 
                            output_dir=os.path.join(PACS_dir,'Vent_Tiled'), 
                            stats_tuple=stats,
                            series_description="Vent_Images")


## -- Tile the GX -- ##
paths = find_file_paths(GX_folder,("image_gas_highreso.nii","membrane.nii","rbc.nii","gas_rgb.npy","membrane2gas_rgb.npy","rbc2gas_rgb.npy"))
nifti_gas = paths["image_gas_highreso.nii"]
nifti_mem = paths["membrane.nii"]
nifti_rbc = paths["rbc.nii"]
numpy_gas = paths["gas_rgb.npy"]
numpy_mem = paths["membrane2gas_rgb.npy"]
numpy_rbc = paths["rbc2gas_rgb.npy"]
GAS = rescale_to_255(nifti_to_numpy(nifti_gas))
MEM = rescale_to_255(nifti_to_numpy(nifti_mem))
RBC = rescale_to_255(nifti_to_numpy(nifti_rbc))
GASrgb = rescale_to_255(np.load(numpy_gas))
MEMrgb = rescale_to_255(np.load(numpy_mem))
RBCrgb = rescale_to_255(np.load(numpy_rbc))
GASrgb = add_colorbar(GASrgb,type='gas')
MEMrgb = add_colorbar(MEMrgb,type='mem')
RBCrgb = add_colorbar(RBCrgb,type='rbc')
GX = tile_arrays_2x3_rgb((GAS,MEM,RBC,GASrgb,MEMrgb,RBCrgb))
dicom_template = values['dicom_template'].replace(os.sep, '/').replace('"','')
output_folder = values['output_directory'].replace(os.sep, '/').replace('"','')
series_description = values['new_series_description'].replace(os.sep, '/').replace('"','')
numpy_to_dicom(numpy_array=GX,
            dicom_template=os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0]),
            output_folder=os.path.join(PACS_dir,'GX_Tiled'),
            series_description='GX_images')


## -- PDF conversion -- ##
convert_pdf_to_dicom(xenoview_pre_pdf,
                     os.path.join(PACS_dir,'Xenoview_pre_pdf'),
                     os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0]),
                     f"Vent_XV_report")

convert_pdf_to_dicom(xenoview_pos_pdf,
                     os.path.join(PACS_dir,'Xenoview_post_pdf'),
                     os.path.join(xenon_pos_folder,os.listdir(xenon_pos_folder)[0]),
                     f"VentBD_XV_report")

convert_pdf_to_dicom(GX_pdf,
                     os.path.join(PACS_dir,'GX_report'),
                     os.path.join(xenon_pos_folder,os.listdir(xenon_pos_folder)[0]),
                     f"GX_report")