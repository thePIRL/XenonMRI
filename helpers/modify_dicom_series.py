## -- modify Dicom series -- ##
import numpy as np
import pydicom
import os
import uuid


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



