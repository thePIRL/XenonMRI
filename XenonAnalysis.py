## -- Code for compliation of data for analysis and export of PACs compatible dicoms  -- ##
import numpy as np
import os
import pydicom
import FlipCal

parent_dir = "c:/tmp"
xenon_pre_folder = "C:/tmp/xenon pre/DICOM/EXP00000/"
xenon_pos_folder = "C:/tmp/xenon post/DICOM/EXP00000"
dicom_pre_template_obj = pydicom.dcmread(os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0]))
dicom_pos_template_obj = pydicom.dcmread(os.path.join(xenon_pre_folder,os.listdir(xenon_pos_folder)[0]))
calibration_twix_path = "C:/tmp/meas_MID00217_FID16268_XE_fid_calibration_dyn.dat"
xenoview_pre_folder = 
xenoview_pos_folder = 
xenoview_pre_pdf = 'pull/from/folder?'
xenoview_pos_pdf = 'pull/from/folder?'
GX_folder = 
GX_pdf = 'pull/from/folder?'

FA = FlipCal.FlipCal(twix_path = calibration_twix_path)
FA.process()
FA.fit_all_DP_FIDs(goFast=True)
FA.process()
FA.completeExport(parent_dir=parent_dir,dummy_dicom_path=os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0]))

