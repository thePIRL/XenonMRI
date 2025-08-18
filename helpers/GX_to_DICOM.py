## - Convert Gas Exchange Images to a set of DICOMS for PACs - ##
import os
import datetime
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from PIL import Image, ImageDraw, ImageFont


def numpy_to_dicom(numpy_array, dicom_template, output_folder, series_description='np_to_DICOM',slice_locations=None,voxel_size = None):
    if voxel_size is None:
        voxel_size = 3.125
    if slice_locations is None:
        slice_locations = np.linspace(-((128 - 1) / 2) * voxel_size,((128 - 1) / 2) * voxel_size,128)
    if isinstance(dicom_template, str):  # - GPT
        dicom_template = pydicom.dcmread(dicom_template)  # - GPT
    def copy_metadata(src_dcm):
        new_dcm = Dataset()
        for elem in src_dcm.iterall():
            if elem.tag != (0x7FE0, 0x0010):  # Exclude Pixel Data
                new_dcm.add(elem)
        return new_dcm
    if numpy_array.ndim != 4 or numpy_array.shape[3] != 3:
        raise ValueError("Input numpy_array must be 4D with shape (rows, cols, slices, rgb)")
    if len(slice_locations) != numpy_array.shape[2]:
        raise ValueError("Length of slice_locations must match the number of slices in numpy_array")
    os.makedirs(f"{output_folder}/", exist_ok=True)
    shared_series_uid = pydicom.uid.generate_uid()
    study_uid = dicom_template.StudyInstanceUID
    for page_num in range(numpy_array.shape[2]):
        image = numpy_array[:, :, page_num, :]
        pixel_array = np.array(image, dtype=np.uint8)
        dicom_file = copy_metadata(dicom_template)
        dicom_file.SOPInstanceUID = pydicom.uid.generate_uid()
        dicom_file.InstanceNumber = page_num + 1
        dicom_file.StudyInstanceUID = study_uid
        dicom_file.SeriesInstanceUID = shared_series_uid
        dicom_file.SeriesNumber = 999
        dicom_file.SeriesDescription = series_description
        dicom_file.ImageType = ["DERIVED", "SECONDARY"]
        dicom_file.ContentDate = datetime.datetime.now().strftime("%Y%m%d")
        dicom_file.ContentTime = datetime.datetime.now().strftime("%H%M%S.%f")[:13]  # HHMMSS.fff
        dicom_file.Manufacturer = "MU PIRL GX_to_DICOM.py v250420"
        dicom_file.SOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
        dicom_file.Rows, dicom_file.Columns, _ = pixel_array.shape
        dicom_file.PhotometricInterpretation = "RGB"
        dicom_file.SamplesPerPixel = 3
        dicom_file.PlanarConfiguration = 0
        dicom_file.BitsAllocated = 8
        dicom_file.BitsStored = 8
        dicom_file.HighBit = 7
        dicom_file.PixelRepresentation = 0
        dicom_file.PixelData = pixel_array.tobytes()
        # Include slice location metadata
        dicom_file.SliceLocation = slice_locations[page_num]
        # Here we indicate what the where the topleft voxel in each image is
        voxel_size = 3.125  # mm 
        dim = 128  # assuming cube 128x128x128 
        x0 = -(dim // 2) * voxel_size  # left-most pixel X coord 
        z0 = (dim // 2) * voxel_size   # top-most pixel Z coord  
        y0 = float(slice_locations[page_num]) 
        dicom_file.ImagePositionPatient = [x0, y0, z0] 
        # Keep consistent orientation unless you want to change it
        # ImageOrientationPatient = [X_row, Y_row, Z_row, X_col, Y_col, Z_col]
        # the patient orientation follows RL=X, AP = Y, FH = Z
        # Don't screw this part up! I always setup numpy arrays with 
        # rows increasing from superior to inferior (Z_row = -1) and
        # columns increasing from right to left (X_col = 1)
        if 'ImageOrientationPatient' not in dicom_file:
            dicom_file.ImageOrientationPatient = [0,0,-1,1,0,0]  # Default Coronal
        dicom_file.is_little_endian = dicom_template.is_little_endian
        dicom_file.is_implicit_VR = dicom_template.is_implicit_VR
        output_path = os.path.join(f"{output_folder}/", f"NPconterved_{page_num:03d}.dcm")
        dicom_file.save_as(output_path)

def rescale_to_255(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr, dtype=np.uint8)
    scaled = 255 * (arr - arr_min) / (arr_max - arr_min)
    return scaled.astype(np.uint8)

import nibabel as nib
def nifti_to_numpy(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata()  # Returns floating point data
    return data


def tile_arrays_2x3_rgb(arrays):
    assert len(arrays) == 6, "Must provide exactly 6 arrays."
    # Normalize all arrays to 4D RGB: shape (128, 128, 128, 3)
    rgb_arrays = []
    for i, arr in enumerate(arrays):
        if arr.ndim == 3 and arr.shape == (128, 128, 128):  # grayscale
            rgb = np.stack([arr]*3, axis=-1)
        elif arr.ndim == 4 and arr.shape == (128, 128, 128, 3):  # RGB
            rgb = arr
        else:
            raise ValueError(f"Array {i+1} has unsupported shape {arr.shape}")
        rgb_arrays.append(rgb)
    # Initialize output array
    output = np.zeros((256, 384, 128, 3), dtype=rgb_arrays[0].dtype)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    for z in range(128):
        # Extract the z-th slice from each array (shape: 128x128x3)
        tiles = [arr[:, :, z, :] for arr in rgb_arrays]
        # Arrange into 2x3 grid
        row1 = np.concatenate(tiles[:3], axis=1)  # shape (128, 384, 3)
        row2 = np.concatenate(tiles[3:], axis=1)  # shape (128, 384, 3)
        stacked = np.concatenate([row1, row2], axis=0)  # shape (256, 384, 3)
        # Assign to output
        pil_img = Image.fromarray(stacked)
        draw = ImageDraw.Draw(pil_img)
        draw.text((50, 230), 'GAS', fill=(255, 255, 255), font=font)
        draw.text((33 + 128, 230), 'MEM/GAS', fill=(255, 255, 255), font=font)
        draw.text((35 + 256, 230), 'RBC/GAS', fill=(255, 255, 255), font=font)
        montage_with_text = np.array(pil_img)
        output[:, :, z, :] = montage_with_text
    return output


def add_colorbar(RGB,type='gas'):
    if type == 'gas' or type == 'rbc':
        for k in range(120,128):
            RGB[k,40:48,:,:] = [255,  0,  0] # - red
            RGB[k,48:56,:,:] = [247,176,  0] # - orange
            RGB[k,56:64,:,:] = [ 97,176, 95] # - dark green
            RGB[k,64:72,:,:] = [  0,248,  1] # - lime green
            RGB[k,72:80,:,:] = [  0,145,176] # - aqua
            RGB[k,80:88,:,:] = [  6, 41,247] # - blue
    elif type == 'mem':
        for k in range(120,128):
            RGB[k,32:40,:,:] = [255,  0,  0] # - red
            RGB[k,40:48,:,:] = [247,176,  0] # - orange
            RGB[k,48:56,:,:] = [ 97,176, 95] # - dark green
            RGB[k,56:64,:,:] = [  0,248,  1] # - lime green
            RGB[k,64:72,:,:] = [184,225,146] # - light green
            RGB[k,72:80,:,:] = [242,205,213] # - light pink
            RGB[k,80:88,:,:] = [224,129,161] # - pink
            RGB[k,88:96,:,:] = [197, 27,124] # - dark pink
    else:
        print("ERROR: Variable 'type' needs to be 'gas', 'mem', or 'rbc'...")
    return RGB


def find_file_paths(directory: str, filenames: tuple) -> dict:
    """
    Search for specific filenames in a directory and return full paths.

    Parameters:
    - directory (str): Path to the directory where the search should occur.
    - filenames (tuple): Tuple of filenames to look for.

    Returns:
    - dict: Dictionary with filenames as keys and full paths as values.
            If a file is not found, the value will be an empty string.
    """
    result = {}
    for fname in filenames:
        full_path = os.path.join(directory, fname)
        result[fname] = full_path if os.path.isfile(full_path) else ''
    return result



if __name__ == '__main__':
    import FreeSimpleGUI as sg
    layout = [
        [sg.Text('Path to Niftis and Numpys:'), sg.Input(key='path_to_data', default_text="", size=(800, 1))],
        [sg.Button('Find my file Paths', key='find_file_paths')],
        [sg.Text('Nifti GAS (gray):'), sg.Input(key='nifti_gas', default_text="image_gas_highreso.nii", size=(800, 1))],
        [sg.Text('Nifti MEM (gray):'), sg.Input(key='nifti_mem', default_text="membrane.nii", size=(800, 1))],
        [sg.Text('Nifti RBC (gray):'), sg.Input(key='nifti_rbc', default_text="rbc.nii", size=(800, 1))],
        [sg.Text('Numpy GAS (RGB) :'), sg.Input(key='numpy_gas', default_text="gas_rgb.npy", size=(800, 1))],
        [sg.Text('Numpy MEM (RGB) :'), sg.Input(key='numpy_mem', default_text="membrane2gas_rgb.npy", size=(800, 1))],
        [sg.Text('Numpy RBC (RGB) :'), sg.Input(key='numpy_rbc', default_text="rbc2gas_rgb.npy", size=(800, 1))],
        [sg.VerticalSeparator(color='blue',pad = 12)],
        [sg.Text('Output Directory (will create if needed):'), sg.Input(key='output_directory', size=(800, 1))],
        [sg.Text('Dicom Template:'), sg.Input(key='dicom_template', size=(800, 1))],
        [sg.Text('New Series Description:'), sg.Input(key='new_series_description', size=(800, 1))],
        [sg.Button('Convert', key='run')],
        [sg.Text("I'm ready to convert a PDF to DICOM using the template's header info.", key='text')]
    ]

    window = sg.Window('Convert a PDF to a DICOM for PACs', layout, return_keyboard_events=False, margins=(0, 0), finalize=True, size=(1000, 500))

    while True:
        event, values = window.read()  # read the window values
        if event == sg.WIN_CLOSED:
            break
        elif event == ('find_file_paths'):
            paths = find_file_paths(values['path_to_data'],("image_gas_highreso.nii","membrane.nii","rbc.nii","gas_rgb.npy","membrane2gas_rgb.npy","rbc2gas_rgb.npy"))
            window['nifti_gas'].update(paths["image_gas_highreso.nii"])
            window['nifti_mem'].update(paths["membrane.nii"])
            window['nifti_rbc'].update(paths["rbc.nii"])
            window['numpy_gas'].update(paths["gas_rgb.npy"])
            window['numpy_mem'].update(paths["membrane2gas_rgb.npy"])
            window['numpy_rbc'].update(paths["rbc2gas_rgb.npy"])
        elif event == ('run'):
            nifti_gas = values["nifti_gas"].replace(os.sep, '/').replace('"','')
            nifti_mem = values["nifti_mem"].replace(os.sep, '/').replace('"','')
            nifti_rbc = values["nifti_rbc"].replace(os.sep, '/').replace('"','')
            numpy_gas = values["numpy_gas"].replace(os.sep, '/').replace('"','')
            numpy_mem = values["numpy_mem"].replace(os.sep, '/').replace('"','')
            numpy_rbc = values["numpy_rbc"].replace(os.sep, '/').replace('"','')
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
                        dicom_template=dicom_template,
                        output_folder=output_folder,
                        series_description=series_description)



