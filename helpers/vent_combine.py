## - Convert Gas Exchange Images to a set of DICOMS for PACs - ##
import os
import datetime
import numpy as np
import pydicom
from pydicom.dataset import Dataset

import os
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from PIL import Image
from copy import deepcopy
from pathlib import Path

def rescale_to_uint8(arr):
    """Rescale a NumPy array to 0–255 uint8."""
    arr = arr.astype(np.float32)
    arr -= arr.min()
    arr /= arr.max() if arr.max() != 0 else 1
    return (arr * 255).astype(np.uint8)
from PIL import Image, ImageDraw, ImageFont

def tile_and_save_rgb_dicom_2x8(input_dirs, output_dir, stats_tuple, series_description = 'Vent_Images'):
    """
    Extended version that adds stats text to the blank column of each RGB DICOM slice.
    
    stats_tuple should be:
    (
        patient_name,
        study_date,
        pre_lung_vol, pre_defect_vol, pre_vdp_ma, pre_vdp_xenoview,
        post_lung_vol, post_defect_vol, post_vdp_ma, post_vdp_xenoview
    )
    """
    assert len(stats_tuple) == 10, "stats_tuple must contain exactly 10 strings."
    os.makedirs(output_dir, exist_ok=True)

    # Unpack stats
    (patient_name, study_date,
     pre_lung_vol, pre_defect_vol, pre_vdp_ma, pre_vdp_xenoview,
     post_lung_vol, post_defect_vol, post_vdp_ma, post_vdp_xenoview) = stats_tuple

    image_series = []

    for i, dir_path in enumerate(input_dirs):
        files = sorted([f for f in Path(dir_path).iterdir() if f.is_file()])
        if not files:
            raise FileNotFoundError(f"No files found in directory: {dir_path}")
        slices = [pydicom.dcmread(f,force = True) for f in files]

        rgb_slices = []
        for ds in slices:
            if not hasattr(ds.file_meta, 'TransferSyntaxUID'):
                ds.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            pixel = ds.pixel_array
            if pixel.ndim == 2:
                pixel = rescale_to_uint8(pixel)
                pixel = np.stack([pixel] * 3, axis=-1)
            elif pixel.ndim == 3 and pixel.shape[2] == 3:
                pixel = pixel.astype(np.uint8)
            else:
                raise ValueError(f"Unsupported pixel shape: {pixel.shape}")
            rgb_slices.append(pixel)
        image_series.append(np.stack(rgb_slices, axis=2))  # shape (H, W, N, 3)

    shapes = [arr.shape for arr in image_series]
    base_shape = shapes[0]
    assert all(s == base_shape for s in shapes), "All series must match in shape"

    H, W, N, _ = base_shape
    stats_width = W  # keep blank column same width
    blank_column = np.zeros((2 * H, stats_width, 3, N), dtype=np.uint8)
    full_stack = []

    # Set font for PIL
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Lines to write
    lines = [
        f"Patient: {patient_name}",
        f"Date: {study_date}",
        "Pre-BD:",
        f"  Lung Vol: {pre_lung_vol}",
        f"  Defect Vol: {pre_defect_vol}",
        f"  VDP (MA): {pre_vdp_ma}",
        f"  VDP (Xenoview): {pre_vdp_xenoview}",
        "Post-BD:",
        f"  Lung Vol: {post_lung_vol}",
        f"  Defect Vol: {post_defect_vol}",
        f"  VDP (MA): {post_vdp_ma}",
        f"  VDP (Xenoview): {post_vdp_xenoview}",
    ]

    # Combine slices and draw text
    for z in range(N):
        s1 = image_series[0][:, :, z, :]
        s2 = image_series[1][:, :, z, :]
        s3 = image_series[2][:, :, z, :]
        s4 = image_series[3][:, :, z, :]
        s5 = image_series[4][:, :, z, :]
        s6 = image_series[5][:, :, z, :]

        col1 = np.vstack([s1, s2])
        col2 = np.vstack([s3, s4])
        col3 = np.vstack([s5, s6])
        montage = np.hstack([blank_column[:, :, :, z], col1, col2, col3])  # (2H, 4W, 3)

        # Draw text into blank column
        pil_img = Image.fromarray(montage)
        draw = ImageDraw.Draw(pil_img)
        y_offset = 40
        for line in lines:
            draw.text((20, y_offset), line, fill=(255, 255, 255), font=font)
            y_offset += 30

        montage_with_text = np.array(pil_img)
        full_stack.append(montage_with_text)

    output_array = np.stack(full_stack, axis=2)

    template = pydicom.dcmread(sorted([f for f in Path(input_dirs[0]).iterdir() if f.is_file()])[0])
    series_uid = template.SeriesInstanceUID  # reuse existing Series UID
    study_uid = template.StudyInstanceUID    # reuse existing Study UID
    for z in range(N):
        new_ds = deepcopy(template)
        rgb_img = output_array[:, :, z, :]
        new_ds.Rows, new_ds.Columns = rgb_img.shape[0], rgb_img.shape[1]
        new_ds.SamplesPerPixel = 3
        new_ds.PhotometricInterpretation = "RGB"
        new_ds.PlanarConfiguration = 0
        new_ds.BitsAllocated = 8
        new_ds.BitsStored = 8
        new_ds.HighBit = 7
        new_ds.PixelRepresentation = 0
        new_ds.PixelData = rgb_img.tobytes()
        new_ds.SOPInstanceUID = generate_uid()
        new_ds.SeriesInstanceUID = series_uid
        new_ds.StudyInstanceUID = study_uid
        new_ds.SeriesDescription = series_description
        new_ds.InstanceNumber = z + 1

        out_path = os.path.join(output_dir, f"tiled_rgb_{z:03d}.dcm")
        new_ds.save_as(out_path)

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


# dirs = (
#     "C:/tmp/Analysis/pre_xenon/DICOM/EXP00000/",  # Top left
#     "C:/tmp/Analysis/post_xenon/DICOM/EXP00000/",  # Bottom left
#     "C:/tmp/Analysis/VentAnalysis_RPT_250413_pre/defectDICOMS/",   # Top middle
#     "C:/tmp/Analysis/VentAnalysis_RPT_250413_post/defectDICOMS/",   # Bottom middle
#     "C:/tmp/Analysis/xenoview PRE/anatomical-ventilation-map/anatomical-ventilation-map-volume/",   # Top right
#     "C:/tmp/Analysis/xenoview POST/anatomical-ventilation-map/anatomical-ventilation-map-volume/"    # Bottom right
# )

# stats = (
#     "Robby Thomen",
#     "2025-04-21",
#     "4.8 L", "0.8 L", "13%", "11%",
#     "5.0 L", "0.6 L", "9%", "7%"
# )

# series_description = 'Robby is cool'

# tile_and_save_rgb_dicom_2x8(dirs, output_dir="C:/tmp/Analysis/Ventilation_compiled", stats_tuple=stats,series_description=series_description)




if __name__ == '__main__':
    import FreeSimpleGUI as sg
    layout = [
        [sg.Text('Pre  - RAW     :'), sg.Input(key='rawpre', default_text="C:/tmp/Analysis/pre_xenon/DICOM/EXP00000/", size=(800, 1))],
        [sg.Text('Post - RAW     :'), sg.Input(key='rawpos', default_text="C:/tmp/Analysis/post_xenon/DICOM/EXP00000/", size=(800, 1))],
        [sg.Text('Pre  - VDP     :'), sg.Input(key='vdppre', default_text="C:/tmp/Analysis/VentAnalysis_RPT_250413_pre/defectDICOMS/", size=(800, 1))],
        [sg.Text('Post - VDP     :'), sg.Input(key='vdppos', default_text="C:/tmp/Analysis/VentAnalysis_RPT_250413_post/defectDICOMS/", size=(800, 1))],
        [sg.Text('Pre  - Xenoview:'), sg.Input(key='xenpre', default_text="C:/tmp/Analysis/xenoview PRE/anatomical-ventilation-map/anatomical-ventilation-map-volume/", size=(800, 1))],
        [sg.Text('Post - Xenoview:'), sg.Input(key='xenpos', default_text="C:/tmp/Analysis/xenoview POST/anatomical-ventilation-map/anatomical-ventilation-map-volume/", size=(800, 1))],
        [sg.VerticalSeparator(color='blue',pad = 12)],
        [sg.Text('Patient Nane      :'), sg.Input(key='ptname', size=(800, 1))],
        [sg.Text('Study Date        :'), sg.Input(key='studydate', size=(800, 1))],
        [sg.Text('Pre Lung Vol      :'), sg.Input(key='prelungvol', size=(800, 1))],
        [sg.Text('Pre Defect Vol    :'), sg.Input(key='predefectvol', size=(800, 1))],
        [sg.Text('Pre VDP           :'), sg.Input(key='prevdp', size=(800, 1))],
        [sg.Text('Pre VDP (Xenoview):'), sg.Input(key='prevdpxen', size=(800, 1))],
        [sg.Text('Post Lung Vol      :'), sg.Input(key='poslungvol', size=(800, 1))],
        [sg.Text('Post Defect Vol    :'), sg.Input(key='posdefectvol', size=(800, 1))],
        [sg.Text('Post VDP           :'), sg.Input(key='posvdp', size=(800, 1))],
        [sg.Text('Post VDP (Xenoview):'), sg.Input(key='posvdpxen', size=(800, 1))],
        [sg.VerticalSeparator(color='blue',pad = 12)],
        [sg.Text('Output Directory (will create if needed):'), sg.Input(key='output_directory', default_text="C:/tmp/Analysis/Ventilation_compiled", size=(800, 1))],
        [sg.Button('Convert', key='run')],
        [sg.Text("I'm ready to compile some ventilation images!...", key='text')]
    ]

    window = sg.Window('Combine all ventilation images', layout, return_keyboard_events=False, margins=(0, 0), finalize=True, size=(1000, 600))

    while True:
        event, values = window.read()  # read the window values
        if event == sg.WIN_CLOSED:
            break
        elif event == ('run'):
            window['text'].update('Compiling...')
            rawpre = values["rawpre"].replace(os.sep, '/').replace('"','')
            vdppre = values["vdppre"].replace(os.sep, '/').replace('"','')
            xenpre = values["xenpre"].replace(os.sep, '/').replace('"','')
            rawpos = values["rawpos"].replace(os.sep, '/').replace('"','')
            vdppos = values["vdppos"].replace(os.sep, '/').replace('"','')
            xenpos = values["xenpos"].replace(os.sep, '/').replace('"','')
            dirs = (rawpre, rawpos, vdppre, vdppos, xenpre, xenpos)
            window['text'].update('Paths tuple created...')
            stats = (
                values['ptname'],
                values['studydate'],
                values['prelungvol'],
                values['predefectvol'],
                values['prevdp'],
                values['prevdpxen'],
                values['poslungvol'],
                values['posdefectvol'],
                values['posvdp'],
                values['posvdpxen'],
            )
            window['text'].update('Stats tuple created...')


            output_folder = values['output_directory'].replace(os.sep, '/').replace('"','')
            window['text'].update('Output directory created...')
            tile_and_save_rgb_dicom_2x8(dirs, output_dir=output_folder, stats_tuple=stats,series_description="Vent_Images")
            window['text'].update(f'Your images were saved to {output_folder}...')



