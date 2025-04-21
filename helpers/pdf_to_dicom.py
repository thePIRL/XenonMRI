import os
import matplotlib.pyplot as plt
import pydicom
import numpy as np
from pydicom.dataset import Dataset
from pdf2image import convert_from_path
from PIL import Image
import datetime

def convert_pdf_to_dicom(pdf_path, output_folder, dicom_template_path, series_description="Converted PDF"):
    """
    Converts each page of a color PDF into a separate DICOM file using metadata from an existing DICOM file.
    I use this to convert the PDF printouts from Xenoview into their own DICOM for PACs upload

    Parameters:
    - pdf_path: str, path to the input PDF file
    - output_folder: str, folder where the output DICOM files will be saved
    - dicom_template_path: str, path to a DICOM file to copy metadata from
    - series_description: str, custom label for the SeriesDescription in the DICOM header

    Outputs:
    - Saves one DICOM file per page in the output folder, ensuring they remain in the same PACS study and series.
    """

    # Load the template DICOM file
    dicom_template = pydicom.dcmread(dicom_template_path)

    # Extract patient and scan metadata
    def copy_metadata(src_dcm):
        new_dcm = Dataset()
        for elem in src_dcm.iterall():
            if elem.tag not in [0x7FE00010]:  # Exclude Pixel Data
                new_dcm.add(elem)
        return new_dcm

    # Convert PDF pages to images
    images = convert_from_path(pdf_path)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Generate a single SeriesInstanceUID to ensure all images stay in the same series
    shared_series_uid = pydicom.uid.generate_uid()

    for page_num, image in enumerate(images, start=1):
        # Convert image to RGB (ensure 24-bit)
        image = image.convert("RGB")  
        pixel_array = np.array(image, dtype=np.uint8)

        # Create a new DICOM file with copied metadata
        dicom_file = copy_metadata(dicom_template)

        # Assign unique values
        dicom_file.SOPInstanceUID = pydicom.uid.generate_uid()  # Unique for each image
        dicom_file.InstanceNumber = page_num  # Order the pages correctly

        # Keep these the same for all pages
        dicom_file.StudyInstanceUID = dicom_template.StudyInstanceUID  # Same study
        dicom_file.SeriesInstanceUID = shared_series_uid  # Same series for all pages
        dicom_file.SeriesNumber = 999  # Consistent series number
        dicom_file.SeriesDescription = series_description  # Custom label
        dicom_file.ImageType = ["DERIVED", "SECONDARY"]
        dicom_file.ContentDate = datetime.datetime.now().strftime("%Y%m%d")
        dicom_file.ContentTime = datetime.datetime.now().strftime("%H%M%S")
        dicom_file.Manufacturer = "MU PIRL pdf_to_dicom.py v250312"
        dicom_file.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage

        # Set DICOM image properties for color
        dicom_file.Rows, dicom_file.Columns, _ = pixel_array.shape
        dicom_file.PhotometricInterpretation = "RGB"
        dicom_file.SamplesPerPixel = 3
        dicom_file.PlanarConfiguration = 0  # RGB pixel arrangement
        dicom_file.BitsAllocated = 8
        dicom_file.BitsStored = 8
        dicom_file.HighBit = 7
        dicom_file.PixelRepresentation = 0
        dicom_file.PixelData = pixel_array.tobytes()

        # Set transfer syntax properties
        dicom_file.is_little_endian = dicom_template.is_little_endian
        dicom_file.is_implicit_VR = dicom_template.is_implicit_VR

        # Save the new DICOM file
        output_path = os.path.join(output_folder, f"page_{page_num:03d}.dcm")
        dicom_file.save_as(output_path)

        print(f"Saved: {output_path}")

    print("Conversion complete.")



# # --- Create DICOMs of the Xenoview PDF Report --- #
# input_pdf = "C:/tmp/ClinicalPatient0029_FCL_XenoviewPOST.pdf"
# output_folder = "C:/tmp/XenoviewPOSTpdf"
# dicom_template = "C:/tmp/post_xenon/DICOM/EXP00000/EXP0009"
# new_series_description = "VentBD_Xenoview_Report" # - What do you want the series to be called in PACs
# convert_pdf_to_dicom(input_pdf,output_folder,dicom_template,new_series_description)



# # --- Create DICOMs of the Gas Exchange Report --- #
# input_pdf = "C:/xenon-gas-exchange-consortium_GabyEdits_250224/assets/Clinical0029_FCL/gx/Lawrence, Frances C_report.pdf"
# output_folder = "C:/xenon-gas-exchange-consortium_GabyEdits_250224/assets/Clinical0029_FCL"
# dicom_template = "C:/tmp/post_xenon/DICOM/EXP00000/EXP0009"
# new_series_description = "GX_Report" # - What do you want the series to be called in PACs
# convert_pdf_to_dicom(input_pdf,output_folder,dicom_template,new_series_description)


if __name__ == '__main__':
    import PySimpleGUI as sg
    layout = [
        [sg.Text('Input PDF:'), sg.Input(key='input_pdf', default_text="", size=(800, 1))],
        [sg.Text('Output Directory (will create if needed):'), sg.Input(key='output_directory', default_text="C:/Users/rptho/Pictures/newnames2/", size=(800, 1))],
        [sg.Text('Dicom Template:'), sg.Input(key='dicom_template', default_text="C:/Users/rptho/Pictures/newnames2/", size=(800, 1))],
        [sg.Text('New Series Description:'), sg.Input(key='new_series_description', default_text="C:/Users/rptho/Pictures/newnames2/", size=(800, 1))],
        [sg.Button('Convert', key='run')],
        [sg.Text("I'm ready to convert a PDF to DICOM using the template's header info.", key='text')]
    ]

    window = sg.Window('Convert a PDF to a DICOM for PACs', layout, return_keyboard_events=False, margins=(0, 0), finalize=True, size=(1000, 300))

    while True:
        event, values = window.read()  # read the window values

        if event == sg.WIN_CLOSED:
            break

        elif event == ('run'):
            input_pdf = values['input_pdf'].replace(os.sep,'/').replace('"','')
            output_folder = values['output_directory'].replace(os.sep,'/').replace('"','')
            dicom_template = values['dicom_template'].replace(os.sep,'/').replace('"','')
            new_series_description = values['new_series_description']
            os.makedirs(output_folder, exist_ok=True)

            convert_pdf_to_dicom(input_pdf,output_folder,dicom_template,new_series_description)

            window['text'].update('Complete!')



# --------------------------------------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------------------------------------- #
def get_slice_locations_from_folder(folder_path):
    """
    Reads all DICOM files in the folder and returns a list of slice locations
    in the order the files were read from the directory.

    Parameters:
        folder_path (str): Path to the folder containing DICOM files.

    Returns:
        list of float or None: Slice locations for each file in the folder.
                               If a file does not contain SliceLocation, None is used.
    """
    slice_locations = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            slice_location = getattr(ds, 'SliceLocation', None)
            slice_locations.append(float(slice_location))
        except Exception as e:
            print(f"Skipping file {filename}: {e}")
            continue
    return slice_locations


import pydicom
import os
import uuid
def modify_dicom_series(input_dir, output_dir, new_series_name, new_series_number=None, change_frame_of_reference=False,SlicLocs = None):
    """
    Modifies DICOM series metadata to ensure it is treated as a separate series in PACS.
    Parameters:
        input_dir (str): Directory containing the original DICOM files.
        output_dir (str): Directory to save modified DICOM files.
        new_series_name (str): New series description name.
        new_series_number (int, optional): Series number to assign (default: auto-increment).
        change_frame_of_reference (bool): Whether to assign a new Frame of Reference UID.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Generate new unique identifiers
    new_series_instance_uid = pydicom.uid.generate_uid()
    new_frame_of_reference_uid = pydicom.uid.generate_uid() if change_frame_of_reference else None
    # Ensure series number is unique (auto-increment if not provided)
    if new_series_number is None:
        new_series_number = int(str(uuid.uuid4().int)[:4])  # Generate a random 4-digit number
    k = 0
    for filename in os.listdir(input_dir):
        filepath = os.path.join(input_dir, filename)
        if not filename.lower().endswith(".dcm"):
            continue  # Skip non-DICOM files
        # Read DICOM file
        dicom_data = pydicom.dcmread(filepath)
        # Modify relevant headers
        dicom_data.SeriesInstanceUID = new_series_instance_uid
        dicom_data.SeriesNumber = new_series_number
        dicom_data.SeriesDescription = new_series_name
        if SlicLocs is not None:
            dicom_data.SliceLocation = SlicLocs[k]
            k=k+1
        if change_frame_of_reference:
            dicom_data.FrameOfReferenceUID = new_frame_of_reference_uid
        # Generate output path
        output_filepath = os.path.join(output_dir, filename)
        # Save modified DICOM file
        dicom_data.save_as(output_filepath)
    print(f"Modified DICOM series saved to {output_dir}")

def rescale_to_255(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = 255 * (arr - arr_min) / (arr_max - arr_min)
    return scaled.astype(np.uint8)

# -- Get slice locations from Folder -- ##
SlicLocs = get_slice_locations_from_folder('C:/tmp/post_xenon/DICOM/EXP00000')


# --- Modify the Xenoview Individual Dicoms with new series description --- #
input_dicom_folder = "C:/tmp/xenoview POST/anatomical-ventilation-map/anatomical-ventilation-map-volume"
output_dicom_folder = "C:/tmp/xenoview POST/anatomical-ventilation-map/RPT"
series_name = "VentBD_Xenoview_VPD=56%_(patient exhaled during scan)"
modify_dicom_series(input_dicom_folder,output_dicom_folder,series_name,SlicLocs=SlicLocs)



