## -- Code for compliation of data for analysis and export of PACs compatible dicoms  -- ##
import numpy as np
import os
import glob
import FlipCal # -- The FlipCal Processing Class
import Vent_Analysis # The Ventilation Processing Class
from helpers.vent_combine import tile_and_save_rgb_dicom_2x8 # -- the function for tiling all vents into 1 image
from helpers.GX_to_DICOM import * # -- Functions for tiling Gas Exchange images into 1 collage
from helpers.pdf_to_dicom import * # -- Function for converting pdfs to DICOMs


def find_my_data(parent_dir):
    '''Thus function will return relavant paths from a single parent directory based on naming conventions.'''
    xenon_pre_folder = ''
    xenon_pos_folder = ''
    proton_pre_folder = ''
    proton_pos_folder = ''
    calibration_twix_path = ''
    xenoview_pre_folder = ''
    xenoview_pos_folder = ''
    xenoview_pre_pdf = ''
    xenoview_pos_pdf = ''
    GX_folder = ''

    xenon_pre_folder = os.path.join(parent_dir,"xenon_pre/DICOM/EXP00000/")
    xenon_pos_folder = os.path.join(parent_dir,"xenon_post/DICOM/EXP00000/")
    proton_pre_folder = os.path.join(parent_dir,"proton_pre/DICOM/EXP00000/")
    if os.path.isdir(os.path.join(parent_dir,"proton_pre/DICOM/EXP00000/")):
        proton_pos_folder = os.path.join(parent_dir,"proton_pre/DICOM/EXP00000/")
    else:
        proton_pos_folder = proton_pre_folder
    calibration_twix_path = glob.glob(os.path.join(parent_dir,'*_dyn.dat'))[0]
    xenoview_pre_folder = os.path.join(parent_dir,"Xenoview_pre")
    xenoview_pos_folder = os.path.join(parent_dir,"Xenoview_post")
    xenoview_pre_pdf = os.path.join(parent_dir,"Xenoview_pre.pdf")
    xenoview_pos_pdf = os.path.join(parent_dir,"Xenoview_post.pdf")
    GX_folder = os.path.join(parent_dir,"GX")
    GX_report = glob.glob(os.path.join(parent_dir,"GX/*_report.pdf"))[0]
    return (xenon_pre_folder,
            xenon_pos_folder,
            proton_pre_folder,
            proton_pos_folder,
            calibration_twix_path,
            xenoview_pre_folder,
            xenoview_pos_folder,
            xenoview_pre_pdf,
            xenoview_pos_pdf,
            GX_folder,
            GX_report)


def PACS_runner(parent_dir,
                xenon_pre_folder,
            xenon_pos_folder,
            proton_pre_folder,
            proton_pos_folder,
            calibration_twix_path,
            xenoview_pre_folder,
            xenoview_pos_folder,
            xenoview_pre_pdf,
            xenoview_pos_pdf,
            GX_folder,
            GX_report,
            xenoview_pre_VDP,
            xenoview_pos_VDP):
    '''You give me a whole buch of data paths (above) and I'll return you a folder of images ready for PACs upload.'''
    
    pre_mask_folder = os.path.join(xenoview_pre_folder, 'lung-segmentation-legacy/lung-segmentation-volume')
    pos_mask_folder = os.path.join(xenoview_pos_folder, 'lung-segmentation-legacy/lung-segmentation-volume')
    print(f"\033[33mPre Mask: {pre_mask_folder} || Post Mask: {pos_mask_folder}\033[37m")

    ## -- Create the PACs upload folder -- ##
    PACS_dir = os.path.join(parent_dir,'PACs')
    os.makedirs(PACS_dir,exist_ok=True)
    print(f"\033[33mPACs directory created: {PACS_dir}\033[37m")    

    dicom_pre_template_path = os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0])
    dicom_pos_template_path = os.path.join(xenon_pos_folder,os.listdir(xenon_pos_folder)[0])

    ## -- Analyze the FlipCal -- ##
    print(f"\033[33mDummy Dicom Paths Pre: {dicom_pre_template_path}, Post: {dicom_pos_template_path}.\033[37m")
    FA = FlipCal.FlipCal(twix_path = calibration_twix_path)
    FA.process(wiggles=False)
    FA.completeExport(parent_dir=parent_dir,dummy_dicom_path=os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0]))
    # FA.dicomPrintout(dummy_dicom_path = dicom_pre_template_path,save_path = os.path.join(PACS_dir,'FlipCal'))
    print(f"\033[33mFlipCal processed successfully.\033[37m")


    ## -- Do the in-house Vent Analysis -- ##
    VentPre = Vent_Analysis.Vent_Analysis(xenon_path=xenon_pre_folder,mask_path=pre_mask_folder,proton_path=proton_pre_folder)
    VentPre.calculate_VDP()
    VentPre.dicom_template_path = dicom_pre_template_path
    VentPre.completeExport(f"{parent_dir}/Vent_pre",dicom_template_path=dicom_pre_template_path,fileName=None,SlicLocs=None,series_description='Vent_printout')
    VentPre.screenShot(path = os.path.join(PACS_dir,'Vent_pre_printout'), series_description = 'Vent_printout')
    print(f"\033[33mVentilation Pre processed successfully.\033[37m")

    VentPos = Vent_Analysis.Vent_Analysis(xenon_path=xenon_pos_folder,mask_path=pos_mask_folder,proton_path=proton_pos_folder)
    VentPos.calculate_VDP()
    VentPos.dicom_template_path = dicom_pos_template_path
    VentPos.completeExport(f"{parent_dir}/Vent_pos",dicom_template_path=dicom_pos_template_path,fileName=None,SlicLocs=None,series_description='VentBD_printout')
    VentPos.screenShot(path = os.path.join(PACS_dir,'Vent_pos_printout'), series_description = 'VentBD_printout')
    print(f"\033[33mVentilation Post processed successfully.\033[37m")


    ## -- Tile the Vents -- ##
    dirs = (xenon_pre_folder, 
            xenon_pos_folder, 
            f"{parent_dir}/Vent_pre/VentDicoms", 
            f"{parent_dir}/Vent_pos/VentDicoms", 
            os.path.join(xenoview_pre_folder,'anatomical-ventilation-map/anatomical-ventilation-map-volume'), 
            os.path.join(xenoview_pos_folder,'anatomical-ventilation-map/anatomical-ventilation-map-volume'))
    stats = (VentPre.metadata['PatientName'],
        VentPre.metadata['StudyDate'],
        np.round(VentPre.metadata['LungVolume'],1),
        np.round(VentPre.metadata['LungVolume']*VentPre.metadata['VDP']/100*1000,1),
        np.round(VentPre.metadata['VDP']),
        xenoview_pre_VDP,
        np.round(VentPos.metadata['LungVolume'],1),
        np.round(VentPos.metadata['LungVolume']*VentPos.metadata['VDP']/100*1000,1),
        np.round(VentPos.metadata['VDP']),
        xenoview_pos_VDP,
    )

    tile_and_save_rgb_dicom_2x8(dirs, 
                                output_dir=os.path.join(PACS_dir,'Vent_Tiled'), 
                                stats_tuple=stats,
                                series_description="Vent_Images")
    print(f"\033[33Ventilation Images tiled and saved successfully.\033[37m")


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
    numpy_to_dicom(numpy_array=GX,
                dicom_template=os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0]),
                output_folder=os.path.join(PACS_dir,'GX_Tiled'),
                series_description='GX_images')
    print(f"\033[33GX Images tiled and saved successfully.\033[37m")


    ## -- PDF conversion -- ##
    convert_pdf_to_dicom(xenoview_pre_pdf,
                        os.path.join(PACS_dir,'Xenoview_pre_pdf'),
                        os.path.join(xenon_pre_folder,os.listdir(xenon_pre_folder)[0]),
                        f"Vent_XV_report")

    convert_pdf_to_dicom(xenoview_pos_pdf,
                        os.path.join(PACS_dir,'Xenoview_post_pdf'),
                        os.path.join(xenon_pos_folder,os.listdir(xenon_pos_folder)[0]),
                        f"VentBD_XV_report")

    convert_pdf_to_dicom(GX_report,
                        os.path.join(PACS_dir,'GX_report'),
                        os.path.join(xenon_pos_folder,os.listdir(xenon_pos_folder)[0]),
                        f"GX_report")
    print(f"\033[32mPDFs converted and saved successfully.\033[37m")
    print(f"\033[32mDone!\033[37m")



## -- GUI -- ##
if __name__ == '__main__':
    import PySimpleGUI as sg
    layout = [
        [sg.Text('Parent Directory:'), sg.Input(key='parent_dir', default_text="c:/tmp/", size=(800, 1))],
        [sg.Button('Populate Fields', key='populate')],
        [sg.Text('Xenon Pre Folder     :'), sg.Input(key='xenon_pre_folder',      size=(800, 1))],
        [sg.Text('Xenon Post Folder    :'), sg.Input(key='xenon_pos_folder',      size=(800, 1))],
        [sg.Text('Proton Pre Folder    :'), sg.Input(key='proton_pre_folder',     size=(800, 1))],
        [sg.Text('Proton Post Folder   :'), sg.Input(key='proton_pos_folder',     size=(800, 1))],
        [sg.Text('Calibration Twix File:'), sg.Input(key='calibration_twix_path', size=(800, 1))],
        [sg.Text('Xenoview Pre Folder  :'), sg.Input(key='xenoview_pre_folder',   size=(800, 1))],
        [sg.Text('Xenoview Post Folder :'), sg.Input(key='xenoview_pos_folder',   size=(800, 1))],
        [sg.Text('Xenoview Pre PDF     :'), sg.Input(key='xenoview_pre_pdf',      size=(800, 1))],
        [sg.Text('Xenoview Post PDF    :'), sg.Input(key='xenoview_pos_pdf',      size=(800, 1))],
        [sg.Text('GX folder            :'), sg.Input(key='GX_folder',             size=(800, 1))],
        [sg.Text('GX PDF report        :'), sg.Input(key='GX_report',             size=(800, 1))],
        [sg.Text('Xenoview Pre VDP     :'), sg.Input(key='xenoview_pre_VDP',      size=(100, 1))],
        [sg.Text('Xenoview Post VDP    :'), sg.Input(key='xenoview_pos_VDP',      size=(100, 1))],
        [sg.Button('PACS my Data!', key='run')],
        [sg.Text("I'm ready to convert a PDF to DICOM using the template's header info.", key='text')]
    ]

    window = sg.Window('Create the PACs images for Xenon Data', layout, return_keyboard_events=False, margins=(0, 0), finalize=True, size=(1000, 500))

    while True:
        event, values = window.read()  # read the window values

        if event == sg.WIN_CLOSED:
            break
        elif event == ('populate'):
            window['text'].update('Populating the Fields...',text_color='yellow')
            xenon_pre_folder,xenon_pos_folder,proton_pre_folder,proton_pos_folder,calibration_twix_path,xenoview_pre_folder,xenoview_pos_folder,xenoview_pre_pdf,xenoview_pos_pdf,GX_folder,GX_report = find_my_data(values['parent_dir'])
            window['xenon_pre_folder'].update(xenon_pre_folder)
            window['xenon_pos_folder'].update(xenon_pos_folder)
            window['proton_pre_folder'].update(proton_pre_folder)
            window['proton_pos_folder'].update(proton_pos_folder)
            window['calibration_twix_path'].update(calibration_twix_path)
            window['xenoview_pre_folder'].update(xenoview_pre_folder)
            window['xenoview_pos_folder'].update(xenoview_pos_folder)
            window['xenoview_pre_pdf'].update(xenoview_pre_pdf)
            window['xenoview_pos_pdf'].update(xenoview_pos_pdf)
            window['GX_folder'].update(GX_folder)
            window['GX_report'].update(GX_report)
            window['text'].update('Fields are populated. Check to ensure they are accurate.',text_color='yellow')
        elif event == ('run'):
            window['text'].update('Assigning variables...',text_color='yellow')
            parent_dir = values['parent_dir']
            xenon_pre_folder = values['xenon_pre_folder']
            xenon_pos_folder = values['xenon_pos_folder']
            proton_pre_folder= values['proton_pre_folder']
            proton_pos_folder= values['proton_pos_folder']
            calibration_twix_path = values['calibration_twix_path']
            xenoview_pre_folder = values['xenoview_pre_folder']
            xenoview_pos_folder = values['xenoview_pos_folder']
            xenoview_pre_pdf = values['xenoview_pre_pdf']
            xenoview_pos_pdf = values['xenoview_pos_pdf']
            GX_folder = values['GX_folder']
            GX_report = values['GX_report']
            xenoview_pre_VDP = values['xenoview_pre_VDP']
            xenoview_pos_VDP = values['xenoview_pos_VDP']
            window['text'].update('Calling PACs runner...',text_color='yellow')
            PACS_runner(parent_dir,
            xenon_pre_folder,
            xenon_pos_folder,
            proton_pre_folder,
            proton_pos_folder,
            calibration_twix_path,
            xenoview_pre_folder,
            xenoview_pos_folder,
            xenoview_pre_pdf,
            xenoview_pos_pdf,
            GX_folder,
            GX_report,
            xenoview_pre_VDP,
            xenoview_pos_VDP)
            window['text'].update('Complete! Your Data has been PACSed!',text_color='#aaffaa')
