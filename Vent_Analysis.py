## -- PIRL Ventilation Image Analysis Pipeline -- ##
import CI # ------------------------------- for calculateCI
import json # ----------------------------- For saving header as json file
import nibabel as nib # ------------------- for Nifti stuffs
import numpy as np
import os
import pickle # --------------------------- For Pickling and unpickling data
from PIL import Image, ImageTk, ImageDraw, ImageFont # ---------- for arrayToImage conversion
import pydicom as dicom # ----------------- for openSingleDICOM and openDICOMFolder
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset # ----- for saving dicoms as PACS in exportDICOM()
from scipy.signal import medfilt2d # ------ for calculateVDP
import SimpleITK as sitk # ---------------- for N4 Bias Correection
import skimage.util # --------------------- for image montages
from sys import getsizeof # --------------- To report twix object size
import time # ----------------------------- for calculateVDP
import datetime # ------------------------- To get the analysis date
import tkinter as tk # -------------------- GUI stuffs
from tkinter import filedialog # ---------- for openSingleDICOM and openDICOMFolder
import mapvbvd # -------------------------- for process_Raw
from sklearn.cluster import KMeans # ---------- for kMeans VDP
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------
# ----------- VENTILATION ANALYSIS CLASS DEFINITION ---------------------------------
#------------------------------------------------------------------------------------
class Vent_Analysis:
    """Performs complete VDP analysis: N4Bias correction, normalization,
        defect calculation, and VDP calculation.
    INPUTS: 
    2 inputs are required at minimum:
        HPvent - 3D array of ventilation image stack
        mask - 3D array of lung segmentation for HPvent (must match HPvent shape)
    these inputs can be called either by direct input as numpy arrays, as paths to dicom files/folders, or as a pickle file to be unpickled
    ATTRIBUTES: (all are declared in __init__ with empty strings, most are self explanatory but heres a list of a few)
        version - Date and author of most recent Vent_Analysis update
        N4HPvent - N4 bias corrected ventilation array
        defectArray - binary array of defect voxels (using mean-anchored array)
        VDP - the ventilation defect percentage using 60% treshold by default (using mean-anchored array)
        CIarray - the Cluster Index Array given the defectArray and vox
        CI - the 95th percentile Cluster Value
        vox - the dimensions of a voxel as a vector
        ds - the complete pydicom object for the HPvent
        metadata - a dictionary of single value properties of the data
    METHODS:
        __init__ - Opens the HP Vent and mask dicoms into self.HPvent, and self.mask
        openSingleDICOM - opens a single 3D dicom file (for vent or proton images)
        openDICOMfolder - opens all 2D dicoms in a folder (for mask images) into 3D array
        pullDICOMHeader - pulls useful DICOM header info into self variables
        calculate_VDP - creates N4HPvent, defectarrays and VDPs
        calculate_CI - calculates CI (imports CI functions)
        exportDICOM - creates a DICOM of the ventilation with defect overlay
        cropToData - using the mask, returns only the rows,cols,slices with signal
        screenShot - Produces a png image of the data
        process_RAW - process the corresponding TWIX file associated
        pickleMe - creates a dictionary of all class attributes and saves as pickle
        unPickleMe - given a pickle, rebuilds the class object
    """
    def __init__(self,xenon_path = None, 
                 mask_path = None, 
                 proton_path = None,
                 xenon_array = None,
                 mask_array=None,
                 proton_array=None,
                 pickle_dict = None,
                 pickle_path = None):
        
        self.version = '250602_vent'
        # 250602 - calculateVDP() now includes Adaptive k-means method (GMGD)
        # 250413 - exportDICOM() now export PACs-compatible dicoms using the xenon dicom header as template
        # 241113 - exportNumpys() added, pickle issues fixed, metadata saves pydicom objects as strings now
        # 241112 - fixed the missing pickle loader
        # 241007 - updated screenshot and unpickling methods
        self.proton = ''
        self.N4HPvent = ''
        self.defectArray = ''
        self.CIarray = ''
        self.vox = ''
        self.ds = ''
        self.twix = ''
        self.raw_k = ''
        self.raw_HPvent = ''
        self.metadata = {'fileName': '',
                        'PatientName': '',
                        'PatientAge': '',
                        'PatientBirthDate' : '',
                        'PatientSex': '',
                        'Disease': '',
                        'StudyDate': '',
                        'SeriesTime': '',
                        'DE': '',
                        'SNR': '',
                        'VDP': '',
                        'VDP_lb': '',
                        'VDP_km': '',
                        'VDP_Akm': '',
                        'LungVolume': '',
                        'DefectVolume': '',
                        'CI': '',
                        'FEV1': '', 
                        'FVC': '',
                        'visit': '',
                        'IRB': '',
                        'treatment': '',
                        'analysisUser': '',
                        'notes': ''
                        # 'TWIXprotocolName': '',
                        # 'TWIXscanDateTime': ''
                        }


        ## ------- XENON DICOM or path ------- ##
        if xenon_array is not None:
            self.HPvent = xenon_array

        if xenon_path is not None:
            try:
                self.ds, self.HPvent = self.openSingleDICOM(xenon_path)
            except:
                try:
                    self.ds, self.HPvent = self.openDICOMfolder(xenon_path)
                except:
                    print('\033[31mOpening Xenon DICOM failed...\033[37m')

            try:
                self.pullDICOMHeader()
            except:
                print('\033[31mPulling Xenon DICOM Header failed...\033[37m')

        ## ------- MASK DICOM FOLDER or path ------- ##
        if mask_array is not None:
            self.mask = mask_array
            self.mask_border = self.calculateBorder(self.mask)

        if mask_path is not None:
            try:
                _, self.mask = self.openDICOMfolder(mask_path)
                self.mask_border = self.calculateBorder(self.mask)
            except:
                print('\033[31mLoading Mask and calculating border failed...\033[37m')


        ## ------- PROTON DICOM or path ------- ##
        if proton_array is not None: 
            self.proton = proton_array

        if proton_path is not None:
            if proton_path is not None: 
                try:
                    self.proton_ds, self.proton = self.openSingleDICOM(proton_path)
                except:
                    try:
                        self.proton_ds, self.proton = self.openDICOMfolder(proton_path)
                    except:
                        print('\033[31mOpening Proton DICOM failed...\033[37m')

        ## ------- PICKLE LOAD ------- ##
        if pickle_path is not None:
            print(f'\033[34mPickle path provided: {pickle_path}. Loading...\033[37m')
            try:
                with open(pickle_path, 'rb') as file:
                    pickle_dict = pickle.load(file)
            except:
                print('\033[31mCould not open Pickle file\033[37m')
        if pickle_dict is not None:
            self.unPickleMe(pickle_dict)
        try:
            self.metadata['LungVolume'] = np.sum(self.mask == 1)*np.prod(np.divide(self.vox,10))/1000
        except Exception as e:
            print(f"Error calculating Lung Volume in __init__() method, probly no vox attribute?: {e}")

        # -- After all data is loaded calcualte lung volume (need mask and vox to do this)
        self.metadata['LungVolume'] = np.sum(self.mask == 1)*np.prod(np.divide(self.vox,10))/1000

        
    def openSingleDICOM(self,dicom_path):        
        if dicom_path is None:
            root = tk.Tk()
            root.withdraw()
            print('\033[94mSelect the DICOM ventilation file...\033[37m')
            dicom_path = tk.filedialog.askopenfilename()
            ds = dicom.dcmread(dicom_path,force=True)
        else:
            ds = dicom.dcmread(dicom_path)
        DICOM_array = ds.pixel_array
        DICOM_array = np.transpose(DICOM_array,(1,2,0))
        print(f'\033[32mI opened a DICOM of shape {DICOM_array.shape}\033[37m')
        return ds, DICOM_array


    def openDICOMfolder(self,maskFolder):  
        from tkinter import filedialog
        if maskFolder is None:
            print('\033[94mSelect the mask folder...\033[37m')
            maskFolder = tk.filedialog.askdirectory()
        dcm_filelist = [f for f in sorted(os.listdir(maskFolder)) if f.endswith('.dcm')]
        if len(dcm_filelist)==0:
            print('No files with .dcm extension. Just pulling all files from folder...')
            dcm_filelist = [f for f in sorted(os.listdir(maskFolder))]
        ds = dicom.dcmread(os.path.join(maskFolder,dcm_filelist[0]))
        mask = np.zeros((ds.pixel_array.shape[0],ds.pixel_array.shape[1],len(dcm_filelist)))
        self.slice_locations = []
        for f,k in zip(dcm_filelist,range(len(dcm_filelist))):
            ds = dicom.dcmread(os.path.join(maskFolder,f))
            mask[:,:,k] = ds.pixel_array
            try:
                slice_location = getattr(ds, 'SliceLocation', None)
                self.slice_locations.append(float(slice_location))
            except:
                print(f'No slice location for image {k}')
        print(f'\033[32mI built a mask of shape {mask.shape}\033[37m')
        return ds, mask

    def pullDICOMHeader(self):
        '''Pulls some of our favorite elements from the DICOM header and adds to metadata dicionary. Voxel dimensions 'vox' are stored separately'''
        infoList = ['PatientName','PatientAge','PatientBirthDate','PatientSize','PatientWeight','PatientSex','StudyDate','StudyTime','SeriesTime']
        for elem in infoList:
            try:
                self.metadata[elem] = self.ds[elem].value
                if 'pydicom' in str(type(self.metadata[elem])):
                    self.metadata[elem] = str(self.metadata[elem]).replace("^","_")
            except:
                print(f'\033[31mNo {elem}\033[37m')
                self.metadata[elem] = ''

        for k in range(100):
            try:
                self.vox = self.ds.PixelSpacing
                break
            except:
                pass

            try:
                self.vox = self.ds[0x5200, 0x9230][k]['PixelMeasuresSequence'][0].PixelSpacing
                break
            except:
                if k == 99:
                    print('Pixel Spacing not in correct place in DICOM header, please enter each dimension (reconstruction dimension, not acquisition!)...')
                    self.vox = [float(input()),float(input())]

        try:
            self.vox = [float(self.vox[0]),float(self.vox[1]),float(self.ds.SpacingBetweenSlices)]
        except:
            print('Slice spacing not in correct position in DICOM header. Please enter manually:')
            self.vox = [float(self.vox[0]),float(self.vox[1]),float(input())]

    def calculateBorder(self,A):
        '''Given a binary array, returns the border of the binary volume (useful for creating a border from the mask for display)'''
        border = np.zeros(A.shape)
        for k in range(A.shape[2]):
            x = np.gradient(A[:,:,k].astype(float))
            border[:,:,k] = (x[0]!=0)+(x[1]!=0)
        return border

    def normalize(self,x):
        if (np.max(x) - np.min(x)) == 0:
            return x
        else:
            return (x - np.min(x)) / (np.max(x) - np.min(x))

    def calculate_VDP(self,thresh=0.6):
        '''Given HPvent and a mask calculates VDPs'''
        self.metadata['SNR'] = self.calculate_SNR(self.HPvent,self.mask) ## -- SNR of xenon DICOM images, not Raw, nor N4
        self.N4HPvent = self.N4_bias_correction(self.HPvent,self.mask)

        ## -- 60% Mean-anchored [Thomen et al. 2015 Radiology] -- ##
        signal_list = sorted(self.N4HPvent[self.mask>0])
        mean_normalized_vent = np.divide(self.N4HPvent,np.mean(signal_list))
        self.defectArray = np.zeros(mean_normalized_vent.shape)
        for k in range(self.mask.shape[2]):
            self.defectArray[:,:,k] = medfilt2d((mean_normalized_vent[:,:,k]<thresh)*self.mask[:,:,k])
        self.defectBorder = self.calculateBorder(self.defectArray) == 1
        self.metadata['VDP'] = 100*np.sum(self.defectArray)/np.sum(self.mask)
        self.metadata['DefectVolume'] = np.sum(self.defectArray == 1)*np.prod(np.divide(self.vox,10))/1000

        ## -- Linear Binning [Mu He, 2016] -- ##
        _99th_percentile_signal_value = signal_list[int(len(signal_list)*.99)]
        norm95th_vent = np.divide(self.N4HPvent,signal_list[int(len(signal_list)*.99)])
        self.defectArrayLB = ((norm95th_vent<=0.16)*1 + (norm95th_vent>0.16)*(norm95th_vent<=0.34)*2 + (norm95th_vent>0.34)*(norm95th_vent<=0.52)*3 + (norm95th_vent>0.52)*(norm95th_vent<=0.7)*4 + (norm95th_vent>0.7)*(norm95th_vent<=0.88)*5 + (norm95th_vent>0.88)*6)*self.mask
        self.metadata['VDP_lb'] = 100*np.sum((self.defectArrayLB == 1)*1 + (self.defectArrayLB == 2)*1)/np.sum(self.mask)



        ## -- Generalized Linear Binning [Mu He, 2020] -- ##
        _99th_percentile_signal_value = signal_list[int(len(signal_list) * 0.99)]
        norm_vent = np.divide(self.N4HPvent, _99th_percentile_signal_value)
        boxcox_thresholds = [0.227, 0.477, 0.686, 0.828, 0.940]         # New non-Gaussian thresholds from Chan et al. (2020)
        self.defectArrayGLB = (                                     #bins
            (norm_vent <= boxcox_thresholds[0]) * 1 +
            ((norm_vent > boxcox_thresholds[0]) & (norm_vent <= boxcox_thresholds[1])) * 2 +
            ((norm_vent > boxcox_thresholds[1]) & (norm_vent <= boxcox_thresholds[2])) * 3 +
            ((norm_vent > boxcox_thresholds[2]) & (norm_vent <= boxcox_thresholds[3])) * 4 +
            ((norm_vent > boxcox_thresholds[3]) & (norm_vent <= boxcox_thresholds[4])) * 5 +
            (norm_vent > boxcox_thresholds[4]) * 6
        ) * self.mask
        self.metadata['VDP_Glb'] = 100 * np.sum((self.defectArrayGLB == 1) + (self.defectArrayGLB == 2)) / np.sum(self.mask)


        ## -- K-Means [Miranda Kirby, 2012] -- ##
        xenon_flattened = self.N4HPvent[self.mask > 0].reshape(-1, 1) 
        KM = KMeans(n_clusters=4, random_state=42)
        labels = KM.fit_predict(xenon_flattened) # perform 4-cluster kMeans on all xenon signal
        cluster_means = [np.mean(xenon_flattened[labels == 0]),np.mean(xenon_flattened[labels == 1]),np.mean(xenon_flattened[labels == 2]),np.mean(xenon_flattened[labels == 3])]
        lowest_cluster = np.argsort(cluster_means)[0] # Which of the 4 clusters has the lowest signal?
        low_xenon = xenon_flattened[labels == lowest_cluster] # make a vector of all low xenon signal
        low_xenon_threshold = np.max(low_xenon) # What is the signal threshold to define hypoventilation
        low_labels = KM.fit_predict(low_xenon) # Do 4-cluster kMeans again on the low cluster (ala Kirby 2013)
        low_cluster_means = [np.mean(low_xenon[low_labels == 0]),np.mean(low_xenon[low_labels == 1]),np.mean(low_xenon[low_labels == 2]),np.mean(low_xenon[low_labels == 3])]
        secondlowest_cluster = np.argsort(low_cluster_means)[1]
        defect_xenon_threshold = np.max(low_xenon[low_labels == secondlowest_cluster])
        self.defectArrayKM = (1*(self.N4HPvent<low_xenon_threshold) + 1*(self.N4HPvent<defect_xenon_threshold))*self.mask
        self.metadata['VDP_km'] = 100*np.sum((self.defectArrayKM >0)*1)/np.sum(self.mask)

        ## -- Adaptive K-Means [Zha, 2016] -- ##
        xenon_flattened = self.N4HPvent[self.mask > 0].reshape(-1, 1)    #extracts all nonzero voxels from the N4image and flattens into a 1D array (vector)
        hist, _ = np.histogram(xenon_flattened, bins=10)                #histogram of vector
        PL = (hist[0] / np.sum(hist)) * 100                             #percentage of lung voxels in the lowest intensity bin so the 1st decile of the histogram
        K1 = 5 if PL < 4 else 4                                         #if PL is <4%, K is 5 if not k=4 becuase fewer lower signal areas exist
        # First round KMeans
        KM1 = KMeans(n_clusters=K1, random_state=42)
        labels1 = KM1.fit_predict(xenon_flattened)                      #assigns cluster label to each voxel
        cluster_means1 = [np.mean(xenon_flattened[labels1 == i]) for i in range(K1)]
        C1_index = np.argmin(cluster_means1)
        C1_voxels = xenon_flattened[labels1 == C1_index]               # voxels in the lower intensity cluster
        # Second round KMeans on C1
        KM2 = KMeans(n_clusters=4, random_state=42)                    #separates into 4 subclusters
        labels2 = KM2.fit_predict(C1_voxels)
        subcluster_means = [np.mean(C1_voxels[labels2 == i]) for i in range(4)]
        sorted_indices = np.argsort(subcluster_means)                   #sorts from lowest to highest signal intensity
        if PL < 4:
            defect_subclusters = sorted_indices[:2]                     #defines number of subclusters depending on PL
        elif 4 <= PL < 10:
            defect_subclusters = sorted_indices[:3]                     #defines number of subclusters depending on PL
        else:
            defect_subclusters = sorted_indices[:1]                     #defines number of subclusters depending on PL
        defect_xenon = C1_voxels[np.isin(labels2, defect_subclusters)]  #calculates mean for each subcluster
        defect_threshold = np.max(defect_xenon)                         
        self.defectArrayAkm = (self.N4HPvent < defect_threshold) * self.mask
        self.metadata['VDP_Akm'] = 100*np.sum(self.defectArrayAkm > 0) / np.sum(self.mask)



        mean_signal = np.mean(self.N4HPvent[self.mask>0])
        self.defect_thresholds = {'MA': thresh,
                                  'LB': _99th_percentile_signal_value*0.34/mean_signal,
                                  'GLB' : _99th_percentile_signal_value*boxcox_thresholds[1]/mean_signal,
                                  'KM': low_xenon_threshold/mean_signal,
                                  'AKM': np.max(C1_voxels) / mean_signal}
        #self.defect_thresholds = [PL, defect_threshold / mean_signal, np.max(C1_voxels) / mean_signal]

        
        print('\033[32mcalculate_VDP ran successfully\033[37m')

    def calculate_CI(self):
        '''Calculates the Cluster Index Array and reports the subject's cluster index (CI)'''
        self.CIarray = CI.calculate_CI(self.defectArray,self.mask,self.vox)
        CVlist = np.sort(self.CIarray[self.defectArray>0])
        index95 = int(0.95*len(CVlist))
        self.metadata['CI'] = CVlist[index95]
        print(f"Calculated CI: {self.metadata['CI']}")

    def exportNifti(self,filepath=None,fileName = None):
        '''Builds all arrays into a 4D array (see build4DdataArrays()) and exports as Nifti'''
        print('\033[34mexportNifti method called...\033[37m')
        if filepath == None:
            print('\033[94mWhere would you like to save your Niftis?\033[37m')
            filepath = tk.filedialog.askdirectory()

        if fileName == None:
            fileName = str(self.metadata['PatientName']).replace('^','_')

        try:
            dataArray = self.build4DdataArray()
            niImage = nib.Nifti1Image(dataArray, affine=np.eye(4))
            savepath = os.path.join(filepath,fileName + '_dataArray.nii')
            nib.save(niImage,savepath)
            print(f'\033[32mNifti HPvent array saved to {savepath}\033[37m')
        except:
            print('\033[31mCould not Export 4D HPvent mask Nifti...\033[37m')

    def build4DdataArray(self):
        ''' Our arrays are: Proton [0], HPvent [1], mask  [2], N4HPvent [3], defectArray [4], CIarray [5]'''
        dataArray = np.zeros((self.HPvent.shape[0],self.HPvent.shape[1],self.HPvent.shape[2],6),dtype=np.float32)
        dataArray[:,:,:,1] = self.HPvent
        dataArray[:,:,:,2] = self.mask
        try:
            dataArray[:,:,:,0] = self.proton
        except:
            print('\033[33mProton either does not exist or does not match Xenon array shape and was not added to 4D array\033[37m')
        try:
            dataArray[:,:,:,3] = self.N4HPvent
        except:
            print('\033[33mN4HPvent does not exist and was not added to 4D array\033[37m')
        try:
            dataArray[:,:,:,4] = self.defectArray
        except:
            print('\033[33mdefectArray does not exist and was not added to 4D array\033[37m')
        try:
            dataArray[:,:,:,5] = self.CIarray
        except:
            print('\033[33mCIarray does not exist and was not added to 4D array\033[37m')
        return dataArray


    def N4_bias_correction(self,HPvent, mask):
        '''Performs N4itk Bias Correction'''
        start_time = time.time()
        print('Performing Bias Correction...')

        # Convert NumPy arrays to SimpleITK images
        image = sitk.GetImageFromArray(HPvent.astype(np.float32))
        mask = sitk.GetImageFromArray(mask.astype(np.float32))

        #Cast to correct format for SimpleITK
        image = sitk.Cast(image, sitk.sitkFloat32)
        mask = sitk.Cast(mask, sitk.sitkUInt8)

        #Run Bias Correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrected_image = corrector.Execute(image, mask)
        corrected_HPvent = sitk.GetArrayFromImage(corrected_image)
        print(f'Bias Correction Completed in {np.round(time.time()-start_time,2)} seconds')
        return corrected_HPvent

    
    def calculate_SNR(self,A,FOVbuffer=20,manualNoise = False):
        '''Calculates SNR using all voxels in the mask as signal, and all 
        voxels oustide the mask bounding box as noise. Can also be done manually if desired'''
        signal = A[self.mask>0]
        if not manualNoise:
            noisemask = np.ones(self.mask.shape)
            FOVbuffer = 20
            rr = (np.sum(np.sum(self.mask,axis = 2),axis = 1)>0)*(list(range(self.mask.shape[0])))
            cc = (np.sum(np.sum(self.mask,axis = 0),axis = 1)>0)*(list(range(self.mask.shape[1])))
            cc = np.arange(np.min(cc[cc>0]),np.max(cc))
            ss = (np.sum(np.sum(self.mask,axis = 1),axis = 0)>0)*(list(range(self.mask.shape[2])))
            noisemask[np.ix_(rr,cc,ss)] = 0
            noisemask[:FOVbuffer,:,:] = 0
            noisemask[(noisemask.shape[0]-FOVbuffer):,:,:] = 0
            noise = A[noisemask==1]
        else:
            pass
            #sub_array = hpg.get_subarray(self.HPvent[:,:,int(self.HPvent.shape[2]/2)])
            #noise = sub_array['A'].flatten()
        SNR = (np.mean(signal)-np.mean(noise))/np.std(noise)
        return SNR
    

    def dicom_to_dict(self, elem, include_private=False):
        '''Given a pydicom object (elem) extracts all elements and builds into dictionary'''
        data_dict = {}
        for sub_elem in elem:
            if not include_private and sub_elem.tag.is_private:
                continue
            if sub_elem.name in ['Pixel Data']:
                continue
            if sub_elem.VR == "SQ":  # Sequence of items
                data_dict[sub_elem.name] = [self.dicom_to_dict(item, include_private) for item in sub_elem.value]
            else:
                data_dict[sub_elem.name] = str(sub_elem.value)
        return data_dict

    def dicom_to_json(self, ds, json_path='c:/pirl/data/DICOMjson.json', include_private=True):
        '''Saves DICOM Header to json file (except the Pixel Data)'''
        dicom_dict = self.dicom_to_dict(ds, include_private)
        with open(json_path, 'w') as json_file:
            json.dump(dicom_dict, json_file, indent=4)
        print(f"\033[32mJson file saved to {json_path}\033[37m")

    def exportNumpys(self,parent_folder = 'c:/pirl/data/'):
        export_path = os.path.join(parent_folder,f'numpys')
        if not os.path.isdir(export_path):
            os.makedirs(export_path)
        try:
            np.save(os.path.join(export_path,'1_HPvent'),self.HPvent)
        except:
            print(f'\033[33mCould not export HPvent to numpy\033[37m')
        try:
            np.save(os.path.join(export_path,'0_proton'),self.proton)
        except:
            print(f'\033[33mCould not export proton to numpy\033[37m')
        try:
            np.save(os.path.join(export_path,'2_mask'),self.mask)
        except:
            print(f'\033[33mCould not export mask to numpy\033[37m')
        try:
            np.save(os.path.join(export_path,'3_N4HPvent'),self.N4HPvent)
        except:
            print(f'\033[33mCould not export N4HPvent to numpy\033[37m')
        try:
            np.save(os.path.join(export_path,'4_defectArray'),self.defectArray)
        except:
            print(f'\033[33mCould not export defectArray to numpy\033[37m')
        try:
            np.save(os.path.join(export_path,'5_CIarray'),self.CIarray)
        except:
            print(f'\033[33mCould not export CIarray to numpy\033[37m')
        try:
            with open(os.path.join(export_path,'metadata.json'), 'w') as json_file:
                json.dump(self.metadata, json_file, indent=4)
        except:
            print(f'\033[33mCould not export metadata json file to numpy\033[37m')
    
    def exportDICOM(self,dicom_template_path,save_dir = 'C:/PIRL/data/',SlicLocs = None,series_description = 'RPT_VentToDICOM'):
        '''Create and saves the Ventilation images with defectArray overlayed'''
        os.makedirs(f"{save_dir}/VentDicoms",exist_ok=True)
        dicom_template = dicom.dcmread(dicom_template_path)
        def copy_metadata(src_dcm):
            new_dcm = Dataset()
            for elem in src_dcm.iterall():
                if elem.tag != (0x7FE0, 0x0010):  # Exclude Pixel Data
                    new_dcm.add(elem)
            return new_dcm
        if self.metadata['VDP'] == '':
            print('\033[31mCant export dicoms until you run calculate_VDP()...\033[37m')
            return
        os.makedirs(save_dir, exist_ok=True)
        BW = (self.normalize(np.abs(self.N4HPvent)) * (2 ** 8 - 1)).astype('uint%d' % 8)
        RGB = np.zeros((self.N4HPvent.shape[0],self.N4HPvent.shape[1],self.N4HPvent.shape[2],3),dtype=np.uint8)
        RGB[:,:,:,0] = BW*(self.defectArray==0) + 255*(self.defectArray==1)
        RGB[:,:,:,1] = BW*(self.defectArray==0)
        RGB[:,:,:,2] = BW*(self.defectArray==0)
        shared_series_uid = dicom.uid.generate_uid()
        study_uid = dicom_template.StudyInstanceUID
        if SlicLocs is None:
            SlicLocs = np.repeat(0,RGB.shape[2])
        for page_num in range(RGB.shape[2]):
            image = RGB[:, :, page_num, :]
            pixel_array = np.array(image, dtype=np.uint8)
            dicom_file = copy_metadata(dicom_template)
            dicom_file.SOPInstanceUID = dicom.uid.generate_uid()
            dicom_file.InstanceNumber = page_num + 1
            dicom_file.StudyInstanceUID = study_uid
            dicom_file.SeriesInstanceUID = shared_series_uid
            dicom_file.SeriesNumber = 999
            dicom_file.SeriesDescription = series_description
            dicom_file.ImageType = ["DERIVED", "SECONDARY"]
            dicom_file.ContentDate = datetime.datetime.now().strftime("%Y%m%d")
            dicom_file.ContentTime = datetime.datetime.now().strftime("%H%M%S.%f")[:13]  # HHMMSS.fff
            dicom_file.Manufacturer = "MU PIRL pdf_to_dicom.py v250312"
            dicom_file.SOPClassUID = dicom.uid.SecondaryCaptureImageStorage
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
            dicom_file.SliceLocation = SlicLocs[page_num]
            # Here we indicate what the where the topleft voxel in each image is
            # voxel_size = 3.125  # mm 
            # dim = 128  # assuming cube 128x128x128 
            # x0 = -(dim // 2) * voxel_size  # left-most pixel X coord 
            # z0 = (dim // 2) * voxel_size   # top-most pixel Z coord  
            # y0 = float(SlicLocs[page_num]) 
            # dicom_file.ImagePositionPatient = [x0, y0, z0] 
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
            dicom_file.save_as(os.path.join(f"{save_dir}/VentDicoms", f"dicom_{page_num:03d}.dcm"))
    
    def completeExport(self,EXPORT_path,dicom_template_path=None,fileName=None,SlicLocs=None,series_description=None):
            os.makedirs(EXPORT_path,exist_ok=True)
            if fileName is None:
                fileName = f"VentAnalysis_{self.metadata['PatientName']}"
            self.exportNumpys(EXPORT_path)
            self.exportNifti(EXPORT_path,fileName)
            self.dicom_to_json(self.ds, json_path=os.path.join(EXPORT_path,f'{fileName}.json'))
            self.pickleMe(pickle_path=os.path.join(EXPORT_path,f'{fileName}.pkl'))
            if series_description is None:
                self.screenShot(os.path.join(EXPORT_path,f'{fileName}.png'),series_description='Vent_printout')
            else:
                self.screenShot(os.path.join(EXPORT_path,f'{fileName}.png'),series_description=series_description)
            if dicom_template_path is not None:
                self.exportDICOM(dicom_template_path=dicom_template_path,
                                save_dir=EXPORT_path,
                                SlicLocs=SlicLocs,
                                series_description=f"Vent_VPD={np.round(self.metadata['VDP'],1)}")
            
    def cropToData(self, A, border=0,borderSlices=False):
        '''Given a 3D mask array, crops rows,cols,slices to only those with signal (useful for creating montage slides)'''
        # Calculate the indices for non-zero slices, rows, and columns
        slices = np.multiply(np.sum(np.sum(A,axis=0),axis=0)>0,list(range(0,A.shape[2])))
        rows = np.multiply(np.sum(np.sum(A,axis=1),axis=1)>0,list(range(0,A.shape[0])))
        cols = np.multiply(np.sum(np.sum(A,axis=2),axis=0)>0,list(range(0,A.shape[1])))
        
        # Filter out the indices for non-zero slices, rows, and columns
        slices = [x for x in range(0,A.shape[2]) if slices[x]]
        rows = [x for x in range(0,A.shape[0]) if rows[x]]
        cols = [x for x in range(0,A.shape[1]) if cols[x]]
        
        # Add border, ensuring we don't exceed the array's original dimensions
        if borderSlices:
            slices_start = max(slices[0] - border, 0)
            slices_end = min(slices[-1] + border + 1, A.shape[2])
        else:
            slices_start = max(slices[0] , 0)
            slices_end = min(slices[-1] + 1, A.shape[2])
        rows_start = max(rows[0] - border, 0)
        rows_end = min(rows[-1] + border + 1, A.shape[0])
        cols_start = max(cols[0] - border, 0)
        cols_end = min(cols[-1] + border + 1, A.shape[1])
        
        # Crop the array with the adjusted indices
        cropped_A = A[rows_start:rows_end, cols_start:cols_end, slices_start:slices_end]
        return cropped_A, list(range(rows_start, rows_end)), list(range(cols_start, cols_end)), list(range(slices_start, slices_end))

    def screenShot(self, path = 'C:/PIRL/data/screenShotTest.png', series_description = 'Vent_printout', normalize95 = False):
        '''Creates and saves a montage image of all processed data images'''
        print(f"\033[33mscreenShot(): called\033[37m")
        def normalize(x):
            if (np.max(x) - np.min(x)) == 0:
                return x
            else:
                return (x - np.min(x)) / (np.max(x) - np.min(x))
        # Load parula colorscale for CI images
        parula = np.load('parula.npy')
        _, rr,cc,ss = self.cropToData(self.mask,border=5)

        # - create the arrays to display from cropped indices
        blank = np.zeros_like(self.HPvent[np.ix_(rr,cc,ss)])
        try:
            proton = normalize(self.proton[np.ix_(rr,cc,ss)])
        except:
            print('No proton array found')
            proton = blank
        HP = normalize(self.HPvent[np.ix_(rr,cc,ss)])
        N4 = normalize(self.N4HPvent[np.ix_(rr,cc,ss)])
        border = normalize(self.mask_border[np.ix_(rr,cc,ss)])>0
        defArr = self.defectArray[np.ix_(rr,cc,ss)]>0
        try:            
            CI = self.CIarray[np.ix_(rr,cc,ss)]
        except:
            CI = blank

        # Create CI color arrays using the parula colorscale
        CIred = np.array([[[parula[int(CI[r,c,s]*64/40),0] for s in range(CI.shape[2])] for c in range(CI.shape[1])] for r in range(CI.shape[0])])
        CIgreen = np.array([[[parula[int(CI[r,c,s]*64/40),1] for s in range(CI.shape[2])] for c in range(CI.shape[1])] for r in range(CI.shape[0])])
        CIblue = np.array([[[parula[int(CI[r,c,s]*64/40),2] for s in range(CI.shape[2])] for c in range(CI.shape[1])] for r in range(CI.shape[0])])

        # The full image is 3 separate 3D montages for each of R G and B
        RED3D = np.concatenate((blank, blank, proton, HP, N4*(~border) + 0*border, N4*(~defArr) + defArr, N4*(CI==0) + CIred*(CI>0)),axis=2)
        GREEN3D = np.concatenate((blank, blank, proton, HP, N4*(~border) + 1*border, N4*(~defArr), N4*(CI==0) +CIgreen*(CI>0)),axis=2)
        BLUE3D = np.concatenate((blank, blank, proton, HP, N4*(~border) + 1*border, N4*(~defArr), N4*(CI==0) +CIblue*(CI>0)),axis=2)
        # Cast the 3D arrays to 2D montages (notice how easy this is with skimage..)
        REDmontage = skimage.util.montage([RED3D[:,:,k] for k in range(0,RED3D.shape[2])], grid_shape = (7,N4.shape[2]), padding_width=0, fill=0)
        GREENmontage = skimage.util.montage([GREEN3D[:,:,k] for k in range(0,GREEN3D.shape[2])], grid_shape = (7,N4.shape[2]), padding_width=0, fill=0)
        BLUEmontage = skimage.util.montage([BLUE3D[:,:,k] for k in range(0,BLUE3D.shape[2])], grid_shape = (7,N4.shape[2]), padding_width=0, fill=0)
        IMAGE = np.stack((REDmontage,GREENmontage,BLUEmontage),axis=2)
        # plt.imshow(IMAGE)
        # plt.show()
        #plt.imsave(path, imageArray) # -- matplotlib command to save array as png

        image = Image.fromarray(np.uint8(IMAGE*255))  # Convert the numpy array to a PIL image
        draw = ImageDraw.Draw(image)
        try:
            for k in ss:
                draw.text((k*N4.shape[1] - N4.shape[1]/2,N4.shape[0]*1.8),f"{k+1}",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 30))
            draw.text((10,N4.shape[0]*0.10), f"Patient: {self.metadata['PatientName']} ({self.metadata['PatientAge']}/{self.metadata['PatientSex']})",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 40))
            draw.text((10,N4.shape[0]*0.40), f"Disease: {self.metadata['Disease']}",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
            draw.text((10,N4.shape[0]*0.70), f"StudyDate: {self.metadata['StudyDate']}",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
            draw.text((10,N4.shape[0]*1.00), f"Visit#: {self.metadata['visit']}",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
            draw.text((10,N4.shape[0]*1.30), f"Treatment: {self.metadata['treatment']}",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
            draw.text((np.round(IMAGE.shape[1]*.25),N4.shape[0]*0.10), f"Lung Volume: {np.round(self.metadata['LungVolume']*1000)} mL",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
            draw.text((np.round(IMAGE.shape[1]*.25),N4.shape[0]*0.40), f"Defect Volume: {np.round(self.metadata['DefectVolume']*1000)} mL",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
            draw.text((np.round(IMAGE.shape[1]*.50),N4.shape[0]*0.10), f"DE: {self.metadata['DE']} mL",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
            draw.text((np.round(IMAGE.shape[1]*.50),N4.shape[0]*0.40), f"FEV1: {self.metadata['FEV1']} %",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
            draw.text((np.round(IMAGE.shape[1]*.50),N4.shape[0]*0.70), f"VDP: {np.round(self.metadata['VDP'], 1)} %",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
        except:
            pass
        try:
            draw.text((np.round(IMAGE.shape[1]*.50),N4.shape[0]*1.00), f"CI: {np.round(self.metadata['CI'])} %",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
        except:
            pass
        try:
            draw.text((np.round(IMAGE.shape[1]*.75),N4.shape[0]*0.25), f'Analysis Version: {self.version}',fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
            draw.text((np.round(IMAGE.shape[1]*.75),N4.shape[0]*0.50), f"Analyzed by: {self.metadata['analysisUser']} on {str(datetime.datetime.today()).split()[0]}",fill = (255,255,255), font = ImageFont.truetype('arial.ttf',size = 35))
        except:
            pass
        image.save(path, 'PNG')  # Save the image
        print(f'\033[32mScreenshot saved to {path}\033[37m')
        try:
            print(f"\033[33mscreenShot(): in dicom creator  {self.dicom_template_path} \n {path}.dcm \033[37m")
            def copy_metadata(src_dcm):
                new_dcm = Dataset()
                for elem in src_dcm.iterall():
                    if elem.tag != (0x7FE0, 0x0010):  # Exclude Pixel Data
                        new_dcm.add(elem)
                return new_dcm
            rgb_array = np.array(image).astype(np.uint8)
            #os.makedirs(f"{output_folder}/", exist_ok=True)
            dicom_template = dicom.dcmread(self.dicom_template_path)
            shared_series_uid = dicom.uid.generate_uid()
            study_uid = dicom_template.StudyInstanceUID
            dicom_file = copy_metadata(dicom_template)
            dicom_file.SOPInstanceUID = dicom.uid.generate_uid()
            dicom_file.InstanceNumber = 1
            dicom_file.StudyInstanceUID = study_uid
            dicom_file.SeriesInstanceUID = shared_series_uid
            dicom_file.SeriesNumber = 999
            dicom_file.SeriesDescription = series_description
            dicom_file.ImageType = ["DERIVED", "SECONDARY"]
            dicom_file.ContentDate = datetime.datetime.now().strftime("%Y%m%d")
            dicom_file.ContentTime = datetime.datetime.now().strftime("%H%M%S.%f")[:13]  # HHMMSS.fff
            dicom_file.Manufacturer = "MU PIRL pdf_to_dicom.py v250312"
            dicom_file.SOPClassUID = dicom.uid.SecondaryCaptureImageStorage
            dicom_file.Rows, dicom_file.Columns, _ = rgb_array.shape
            dicom_file.PhotometricInterpretation = "RGB"
            dicom_file.SamplesPerPixel = 3
            dicom_file.PlanarConfiguration = 0
            dicom_file.BitsAllocated = 8
            dicom_file.BitsStored = 8
            dicom_file.HighBit = 7
            dicom_file.PixelRepresentation = 0
            dicom_file.PixelData = rgb_array.tobytes()
            dicom_file.is_little_endian = dicom_template.is_little_endian
            dicom_file.is_implicit_VR = dicom_template.is_implicit_VR
            dicom_file.save_as(f"{path}.dcm")
        except:
            print('\033[33mExporting printout as DICOM failed...\033[37m')

    def process_RAW(self,filepath=None):
        '''Given a twix file, will extract the raw kSpace and reconstruct the image array.
            3 new attributes are created here from the twix file: the twix object, the raw kSpace array, and the image array (kSpace fft'd).
            I'm not a huge fan of this because having objects as class attributes isn't super backwards compatible, nor convertable across versions.
            This is why when we pickle the Vent_Analysis object, all attributes are converted to dictionaries first (I learned that the hard way).
            Anyways, in the future I think we need to do the same here: extract twix object attributes to a dictionary. But idk. 
            The ISMRMRD can help here, BUT header info is lost in this conversion so I don't want to do it exclusively...'''
        if filepath == None:
            print('\033[94mSelect the corresponding RAW data file (Siemens twix)...\033[37m\n')
            filepath = tk.filedialog.askopenfilename()
        self.raw_twix = mapvbvd.mapVBVD(filepath)
        self.metadata['TWIXscanDateTime'] = self.raw_twix.hdr.Config['PrepareTimestamp']
        self.metadata['TWIXprotocolName'] = self.raw_twix.hdr.Meas['tProtocolName']
        self.raw_twix.image.squeeze = True
        self.raw_K = self.raw_twix.image['']
        self.raw_HPvent = np.zeros((self.raw_K.shape)).astype(np.complex128)
        for k in range(self.raw_K.shape[2]):
            self.raw_HPvent[:,:,k] = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.raw_K[:,:,k])))
        self.raw_HPvent = np.transpose(self.raw_HPvent,(1,0,2))[:,::-1,:]

    def pickleMe(self, pickle_path='C:/PIRL/data/VentPickle.pkl'):
        '''Uses dictionary comprehension to create a dictionary of all class attributes, then saves as pickle'''
        pickle_dict = {}
        for attr in vars(self):
            if 'pydicom' in str(type(attr)):
                print(f'omitting {attr} from pickle')
                continue
            try:
                pickle.dumps(getattr(self, attr))
                pickle_dict[attr] = getattr(self, attr)
            except (pickle.PicklingError, AttributeError, TypeError):
                print(f"\033[31mSkipping non-picklable attribute: {attr}\033[37m")
        with open(pickle_path, 'wb') as file:
            pickle.dump(pickle_dict, file)
        print(f'\033[32mPickled dictionary saved to {pickle_path}\033[37m')
    
    def unPickleMe(self,pickle_dict):
        '''Given a pickled dictionary (yep, I actually named a variable pickle_dict), it will extract entries to class attributes'''
        for attr, value in pickle_dict.items():
            try:
                setattr(self, attr, value)
            except:
                pass
    
    def __repr__(self):
        string = (f'\033[35mVent_Analysis\033[37m class object version \033[94m{self.version}\033[37m\n')
        for attr, value in vars(self).items():
            if value == '':
                string += (f'\033[31m {attr}: \033[37m\n')
            elif type(value) is np.ndarray:
                string += (f'\033[32m {attr}: \033[36m{value.shape} \033[37m\n')
            elif type(value) is dict:
                for attr2, value2 in value.items():
                    if value2 == '':
                        string += (f'   \033[31m {attr2}: \033[37m\n')
                    else:
                        string += (f'   \033[32m {attr2}: \033[36m{value2} \033[37m\n')
            else:
                string += (f'\033[32m {attr}: \033[36m{type(value)} \033[37m\n')
        return string

def extract_attributes(attr_dict, parent_key='', sep='_'):
    """
    Recursively extract all attributes and subattributes from a nested dictionary and compiles into flat dictionary.
    
    Args:
    - attr_dict (dict): The attribute dictionary to extract from.
    - parent_key (str): The base key to use for building key names for subattributes.
    - sep (str): The separator to use between nested keys.
    
    Returns:
    - dict: A flat dictionary with all attributes and subattributes.
    """
    items = []
    for k, v in attr_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            # If the value is a dictionary, recurse
            items.extend(extract_attributes(v, new_key, sep=sep).items())
        else:
            # Otherwise, add the attribute to the items list
            items.append((new_key, v))
    return dict(items)



# # ## -- Test Code -- ##
# DICOM_path = '//umh.edu/data/Radiology/Xenon_Studies/Studies/MEPO/MEPO_Studies/MEPOXE0039 - 240301/Pre-Alb/DICOM/24030116/46420001/48522586_Xe'
# MASK_path = '//umh.edu/data/Radiology/Xenon_Studies/Studies/MEPO/MEPO_Studies/MEPOXE0039 - 240301/Pre-Alb/DICOM/HPImg/Mask'
# PROTON_path = '//umh.edu/data/Radiology/Xenon_Studies/Studies/MEPO/MEPO_Studies/MEPOXE0039 - 240301/Pre-Alb/DICOM/HPImg/48522597'
# Vent1 = Vent_Analysis(proton_path=PROTON_path, xenon_path=DICOM_path, mask_path=MASK_path)
# Vent1.calculate_VDP()

# # ## -- Test Code -- ##
# DICOM_path = 'C:/PIRL/data/MEPOXE0039/48522586xe'
# MASK_path = 'C:/PIRL/data/MEPOXE0039/Mask'
# PROTON_path = 'C:/PIRL/data/MEPOXE0039/48522597prot'
# Vent1 = Vent_Analysis(proton_path=PROTON_path, xenon_path=DICOM_path, mask_path=MASK_path)
# Vent1.calculate_VDP()

# # EXPORT_path = 'C:/PIRL/data/MEPOXE0039/VentAnalysis_RPT_testing/'
# # if not os.path.isdir(EXPORT_path):
# #     os.makedirs(EXPORT_path)

# EXPORT_path = 'C:/PIRL/data/MEPOXE0039/VentAnalysis_RPT_testing/'
# Vent1.exportNifti(EXPORT_path,'nifti.nii')
# Vent1.dicom_to_json(Vent1.ds, json_path=os.path.join(EXPORT_path,f'json.json'))
# Vent1.pickleMe(pickle_path=os.path.join(EXPORT_path,f'pkl.pkl'))
# Vent1.screenShot(path=os.path.join(EXPORT_path,f'png.png'))
# Vent1.exportDICOM(Vent1.ds,EXPORT_path,optional_text='testing',forPACS=True)

# # for attr, value in Vent1.metadata.items():
# #     print(f'{attr} is a {type(value)}')

# Vent2 = Vent_Analysis(pickle_path=os.path.join(EXPORT_path,f'pkl.pkl'))
# Vent2.exportDICOM(Vent1.ds,EXPORT_path,optional_text='testing',forPACS=True)
# Vent2.exportNumpys('C:/PIRL/data/MEPOXE0039/VentAnalysis_RPT_241113')

# Vent3 = Vent_Analysis(pickle_path="//umh.edu/data/Radiology/Xenon_Studies/Studies/General_Xenon/Gen_Xenon_Studies/Xe-0078 - 240213 - pnemonia_lung toxicity_partial lobectomy/Xe-0078_240208_preAlb.pkl")

# with open('C:/PIRL/data/MEPOXE0039/VentAnalysis_RPT_241113/numpys/metadata.json', 'w') as json_file:
#     json.dump(Vent2.metadata, json_file, indent=4)

# Vent1 = Vent_Analysis(pickle_path="C:/tmp/VentAnalysis_RPT_250413/Clinical_FCL_250410_visit1_Albuterol.pkl")
# Vent1.exportDICOM(Vent1.ds,save_dir = 'c:/tmp/',optional_text='RPT2',series_description='Vent_VDP=%')

# def get_slice_locations_from_folder(folder_path):
#     slice_locations = []
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)
#         try:
#             ds = dicom.dcmread(file_path, stop_before_pixels=True)
#             slice_location = getattr(ds, 'SliceLocation', None)
#             slice_locations.append(float(slice_location))
#         except Exception as e:
#             print(f"Skipping file {filename}: {e}")
#             continue
#     return slice_locations

# get_slice_locations_from_folder("C:/tmp/pre_xenon/DICOM/EXP00000")
### ------------------------------------------------------------------------------------------------ ###
### ---------------------------------------- MAIN GUI SCRIPT --------------------------------------- ###
### ------------------------------------------------------------------------------------------------ ###

if __name__ == "__main__":
    version = '250413_VentAnalysisGUI'
    # - 250413 - Now uses DICOM template path to populate the exportDICOM() header so defects images can be put in PACS
    image_box_size = 40
    ARCHIVE_path = '//umh.edu/data/Radiology/Xenon_Studies/Studies/Archive/'
    
    import PySimpleGUI as sg
    from datetime import date # -- So we can export the analysis date

    ## -- Helper Functions for GUI -- ##
    def arrayToImage(A,size):
        imgAr = Image.fromarray(A.astype(np.uint8))
        imgAr = imgAr.resize(size)
        image = ImageTk.PhotoImage(image=imgAr)
        return(image)

    def normalize(x):
        if (np.max(x) - np.min(x)) == 0:
            return x
        else:
            return (x - np.min(x)) / (np.max(x) - np.min(x))

    def colorBinary(A,B):
        A = normalize(A)
        new = np.zeros((A.shape[0],A.shape[1],3))
        new[:,:,0] = A*(B==0) + B
        new[:,:,1] = A*(B==0)
        new[:,:,2] = A*(B==0)
        return new*255

    def colorparula(A,B):
        A = normalize(A)
        new = np.zeros((A.shape[0],A.shape[1],3))
        new[:,:,0] = A*(B==0) + B
        new[:,:,1] = A*(B==0)
        new[:,:,2] = A*(B==0)
        return new*255
    
    def array3D_to_montage2D(A):
        return skimage.util.montage([abs(A[:,:,k]) for k in range(0,A.shape[2])], grid_shape = (1,A.shape[2]), padding_width=0, fill=0)
    
    sg.theme('Default1')
    PIRLlogo = os.path.join(os.getcwd(),'PIRLlogo.png')
    path_label_column = [[sg.Text('Path to Ventilation DICOM:')],[sg.Text('Path to Mask Folder:')],[sg.Text('Path to Proton:')],[sg.Text('Path to Twix:')]]
    path_column = [[sg.InputText(key='DICOMpath',default_text="C:/tmp/86245872xe_pre",size=(200,200))],
                   [sg.InputText(key='MASKpath',default_text="C:/tmp/xenoview PRE/lung-segmentation-legacy/lung-segmentation-volume",size=(200,200))],
                   [sg.InputText(key='PROTONpath',default_text="C:/tmp/86245905ins",size=(200,200))],
                   [sg.InputText(key='TWIXpath',size=(200,200))]]
    
    IRB_select_column = [
                    [sg.Radio('GenXe','IRB',key='genxeRadio',enable_events=True)],
                    [sg.Radio('Mepo','IRB',key='mepoRadio',enable_events=True)],
                    [sg.Radio('Clinical','IRB',key='clinicalRadio',enable_events=True)]]
    genxe_info_column = [[sg.Text('General Xenon ID:'),sg.InputText(default_text='0000',size=(10,10),key='genxeID')],
                           [sg.Text('Disease:'),sg.Radio('Healthy','disease',key='diseaseHealthy'),sg.Radio('Asthma','disease',key='diseaseAsthma'),sg.Radio('CF','disease',key='diseaseCF'),sg.Radio('COPD','disease',key='diseaseCOPD'),sg.Radio('Other:','disease',key='diseaseOther'),sg.InputText(size=(10,1))],
                           [sg.Checkbox('PreAlbuterol',default=False,key='prealb'),sg.Checkbox('PostAlbuterol',default=False,key='postalb'),sg.Checkbox('PreSildenafil',default=False,key='presil'),sg.Checkbox('PostSildenafil',default=False,key='postsil')],
                           ]
    mepo_info_column = [[sg.Text('Mepo ID:'),sg.InputText(default_text='0000',size=(10,10),key='mepoID')],
                        [sg.Text('Mepo Subject #:    '),sg.InputText(default_text='0',size=(10,10),key='meposubjectnumber')],
                        [sg.Text('Visit:    '),sg.Radio('Baseline','mepo_visit',key='mepoVisit1'),sg.Radio('4-week','mepo_visit',key='mepoVisit2'),sg.Radio('12-week','mepo_visit',key='mepoVisit3')],
                        [sg.Radio('PreAlbuterol','mepoalbuterol',key='prealb_mepo'),sg.Radio('PostAlbuterol','mepoalbuterol',key='postalb_mepo')],
                        ]
    clinical_info_column = [[sg.Text('Clinical Subject Initials:'),sg.InputText(default_text='',size=(10,10),key='clinicalID')],
                           [sg.Text('Visit #:    '),sg.InputText(default_text='0',size=(10,10),key='clinicalvisitnumber')],
                           [sg.Radio('Baseline','clinicalalbuterol',key='baseline'),
                            sg.Radio('Albuterol','clinicalalbuterol',key='albuterol')],
                           ]
    dose_info_column = [[sg.Text('DE [mL]:'),sg.InputText(key='DE',size=(10,10))],
                           [sg.Text('FEV1 [%]: '),sg.InputText(key='FEV1',size=(10,10))],
                           [sg.Text('FVC [%]: '),sg.InputText(key='FVC',size=(10,10))],
                           ]

    patient_data_column = [[sg.Button('',key='editPatientName',pad=(0,0)),sg.Text('Subject:                               ',key='subject',pad=(0,0))],
                           [sg.Button('',key='editStudyDate',pad=(0,0)),sg.Text('Study Date:',key='studydate',pad=(0,0))],
                           [sg.Button('',key='editStudyTime',pad=(0,0)),sg.Text('Study Time:',key='studytime',pad=(0,0))],
                           [sg.Button('',key='editTwixDate',pad=(0,0)),sg.Text('Twix Date:',key='twixdate',pad=(0,0))],
                           [sg.Button('',key='editProtocol',pad=(0,0)),sg.Text('Protocol:',key='twixprotocol',pad=(0,0))],
                           [sg.Button('',key='editPatientAge',pad=(0,0)),sg.Text('Age:',key='age',pad=(0,0))],
                           [sg.Button('',key='editPatientSex',pad=(0,0)),sg.Text('Sex:',key='sex',pad=(0,0))],
                           [sg.Button('',key='editPatientDOB',pad=(0,0)),sg.Text('DOB:',key='dob',pad=(0,0))],]
    dicom_data_column = [[sg.Text('DICOM Voxel Size:                                ',key = 'vox',pad=(0,0))],
                         [sg.Text('SNR:',key = 'snr',pad=(0,0))],
                         [sg.Text('VDP:',key = 'vdp',pad=(0,0))],
                         [sg.Text('Ventilation Array Shape:',key='ventarrayshape',pad=(0,0))],
                         [sg.Text('Mask Lung Vol:',key='masklungvol',pad=(0,0))],
                         [sg.Text('Defect Volume:',key='defectvolume',pad=(0,0))],
                         [sg.Text('CI:',key='ci',pad=(0,0))]]
    image_column = [[sg.Image(key='-PROTONIMAGE-')],
                    [sg.Image(key='-RAWIMAGE-')],
                    [sg.Image(key='-N4IMAGE-')],
                    [sg.Image(key='-DEFECTIMAGE-')],
                    [sg.Image(key='-CIIMAGE-')],
                    [sg.Image(key='-TWIXIMAGE-')]]

    layout = [
        [sg.Image(PIRLlogo),sg.Text(f'version {version}'),sg.Text('         User:'),sg.InputText(key='userName',size=(10,1),enable_events=False),sg.Button('-',key='minus'),sg.Button('+',key='plus')],
        [sg.HorizontalSeparator()],
        [sg.Column(path_label_column),sg.Column(path_column)],
        [sg.Button('Load from Paths', key='-INITIALIZE-'),sg.Button('Calculate VDP', key='-CALCVDP-'),sg.Button('Calculate CI', key='-CALCCI-'),sg.Button('Import TWIX', key='-RUNTWIX-'),sg.Button('Load Pickle', key='-LOADPICKLE-',pad = (300,0))],  
        [sg.HorizontalSeparator()],
        [sg.Column(IRB_select_column),
         sg.Column(clinical_info_column,key='clinicalInputs',visible=False),
         sg.Column(genxe_info_column,key='genxeInputs',visible=False),
         sg.Column(mepo_info_column,key='mepoInputs',visible=False),
         sg.Column(dose_info_column)],
        [sg.HorizontalSeparator()],
        [sg.Column(patient_data_column),sg.VSeperator(),sg.Column(dicom_data_column),sg.VSeperator(),sg.Column(image_column)],  
        [sg.Text('Notes:'),sg.InputText(key='notes',size=(200,200))],
        [sg.Text('',key = '-STATUS-')],
        [sg.Text('DICOM Template Path:'),sg.InputText(key='dicom_template_path',size=(200,200))],
        [sg.Text('Export Path:'),sg.InputText(key='exportpath',default_text='C:/tmp/',size=(200,200))],
        [sg.Button('Export Data',key='-EXPORT-'),sg.Checkbox('Copy pickle to Archive',default=True,key='-ARCHIVE-'),sg.Push(),sg.Button('Clear Cache',key='-CLEARCACHE-')]
    ]

    window = sg.Window(f'PIRL Ventilation Analysis -- {version}', layout, return_keyboard_events=True, margins=(0, 0), finalize=True, size= (1200,730),resizable=True)

    def updateImages():
            window['-TWIXIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))
            try:
                protonMontage = array3D_to_montage2D(Vent1.proton)
                protonMontageImage = arrayToImage(255*normalize(protonMontage),(int(image_box_size*protonMontage.shape[1]/protonMontage.shape[0]),image_box_size))
                window['-PROTONIMAGE-'].update(data=protonMontageImage)
            except:
                window['-PROTONIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

            try:
                rawMontage = array3D_to_montage2D(Vent1.HPvent)
                mask_border = array3D_to_montage2D(Vent1.mask_border)
                rawMontageImage = arrayToImage(colorBinary(rawMontage,mask_border),(int(image_box_size*rawMontage.shape[1]/rawMontage.shape[0]),image_box_size))
                window['-RAWIMAGE-'].update(data=rawMontageImage)
            except:
                window['-RAWIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

            try:
                N4Montage = array3D_to_montage2D(Vent1.N4HPvent)
                mask_border = array3D_to_montage2D(Vent1.mask_border)
                N4MontageImage = arrayToImage(colorBinary(N4Montage,mask_border),(int(image_box_size*N4Montage.shape[1]/N4Montage.shape[0]),image_box_size))
                window['-N4IMAGE-'].update(data=N4MontageImage)
            except:
                window['-N4IMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

            try:
                DefectMontage = array3D_to_montage2D(Vent1.defectArray)
                DefectMontageImage = arrayToImage(colorBinary(N4Montage,DefectMontage),(int(image_box_size*N4Montage.shape[1]/N4Montage.shape[0]),image_box_size))
                window['-DEFECTIMAGE-'].update(data=DefectMontageImage)
            except:
                window['-DEFECTIMAGE-'].update(data=arrayToImage(np.zeros((3,3)),(1000,image_box_size)))

            try:
                CIMontage = array3D_to_montage2D(Vent1.CIarray)
                CIMontageImage = arrayToImage(colorBinary(N4Montage, CIMontage), (int(image_box_size * N4Montage.shape[1] / N4Montage.shape[0]), image_box_size))
                window['-CIIMAGE-'].update(data=CIMontageImage)
            except:
                window['-CIIMAGE-'].update(data=arrayToImage(np.zeros((3, 3)), (1000, image_box_size)))


    def updateData():
        if 'Vent1' in globals():
            window['subject'].update(f"Subject: {Vent1.metadata['PatientName']}")
            window['studydate'].update(f"Study Date: {Vent1.metadata['StudyDate']}")
            window['studytime'].update(f"Study Time: {Vent1.metadata['StudyTime']}")
            window['age'].update(f"Age: {Vent1.metadata['PatientAge']}")
            window['sex'].update(f"Sex: {Vent1.metadata['PatientSex']}")
            window['dob'].update(f"DOB: {Vent1.metadata['PatientBirthDate']}")
            window['vox'].update(f"DICOM voxel Size: {Vent1.vox} [mm]")
            window['snr'].update(f"SNR: {Vent1.metadata['SNR']}")
            window['vdp'].update(f"VDP: {Vent1.metadata['VDP']}")
            window['ventarrayshape'].update(f'Ventilation Array Shape: {Vent1.HPvent.shape}')
            window['masklungvol'].update(f"Mask Lung Volume: {str(Vent1.metadata['LungVolume'])} [L]")
            try:
                window['defectvolume'].update(f"Defect Volume: {str(Vent1.metadata['DefectVolume'])} [L]")
                window['ci'].update(f"CI: {Vent1.metadata['CI']}")
            except:
                pass
            window['twixdate'].update(f'Twix Date:')
            window['twixprotocol'].update(f'Twix Protocol:')

    updateImages()
    updateData()

    while True:
        event, values = window.read()
        #print("")
        #print(event)
        #print(values)
        if event == sg.WIN_CLOSED:
            break
## --------------- PLUS MINUS BUTTONS --------------------------- ##
        elif event == ('minus'):
            image_box_size = image_box_size-5
            updateImages()
        elif event == ('plus'):
            image_box_size = image_box_size+5
            updateImages()

## --------------- STUDY SELECT RADIO BUTTONS ------------------- ##
        elif event == ('mepoRadio'):
            IRB = 'Mepo'
            window['genxeInputs'].update(visible=False)
            window['mepoInputs'].update(visible=True)
            window['clinicalInputs'].update(visible=False)
        elif event == ('genxeRadio'):
            IRB = 'GenXe'
            window['genxeInputs'].update(visible=True)
            window['mepoInputs'].update(visible=False)
            window['clinicalInputs'].update(visible=False)
        elif event == ('clinicalRadio'):
            IRB = 'Clinical'
            window['genxeInputs'].update(visible=False)
            window['mepoInputs'].update(visible=False)
            window['clinicalInputs'].update(visible=True)

## --------------- Info Edit Buttons ------------------- ##
        elif event == ('editPatientName'):
            text = sg.popup_get_text('Enter Subject ID: ')
            window['subject'].update(f'Subject: {text}')
            Vent1.metadata['PatientName'] = text
        elif event == ('editPatientAge'):
            text = sg.popup_get_text('Enter Patient Age: ')
            window['age'].update(f'Age: {text}')
            Vent1.metadata['PatientAge'] = text
        elif event == ('editPatientSex'):
            text = sg.popup_get_text('Enter Patient Sex: ')
            window['sex'].update(f'Sex: {text}')
            Vent1.metadata['PatientSex'] = text
        elif event == ('editPatientDOB'):
            text = sg.popup_get_text('Enter Patient DOB: ')
            window['dob'].update(f'DOB: {text}')
            Vent1.metadata['PatientDOB'] = text

## --------------- Load Pickle ------------------- ##       
        elif event == ('-LOADPICKLE-'):
            pickle_path = sg.popup_get_text('Enter Pickle Path: ',default_text="C:/PIRL/data/MEPOXE0039/VentAnalysis_RPT_241006/Mepo0039_240301.pkl").replace('"','')
            Vent1 = Vent_Analysis(pickle_path=pickle_path)
            window['-STATUS-'].update("Vent_Analysis pickle loaded",text_color='green')
            window['-INITIALIZE-'].update(button_color = 'green')
            updateData()
            updateImages()

## --------------- INITIALIZE Button ------------------- ##
        elif event == ('-INITIALIZE-'):
            DICOM_path = values['DICOMpath'].replace('"','')
            MASK_path = values['MASKpath'].replace('"','')
            TWIX_path = values['TWIXpath'].replace('"','')
            PROTON_path = values['PROTONpath'].replace('"','')
            window['-CALCVDP-'].update(button_color = 'lightgray')
            window['-CALCCI-'].update(button_color = 'lightgray')
            window['-RUNTWIX-'].update(button_color = 'lightgray')
            window['-EXPORT-'].update(button_color = 'lightgray')
            try:
                del Vent1
                print('cleared Vent1')
            except:
                print('cache already clean')
            try:
                if PROTON_path == '':
                    Vent1 = Vent_Analysis(DICOM_path,MASK_path)
                else:
                    Vent1 = Vent_Analysis(DICOM_path,MASK_path,PROTON_path)
                window['-STATUS-'].update("Vent_Analysis loaded",text_color='green')
                window['-INITIALIZE-'].update(button_color = 'green')
                updateData()
                updateImages()
            except:
                window['-STATUS-'].update("ERROR: Uhh you messed something up. Maybe check your DICOM and MASK paths?",text_color='red')
                continue

## --------------- CALCULATE VDP Button ------------------- ##
        elif event == ('-CALCVDP-'):
            try:
                window['-STATUS-'].update("Calculating VDP...",text_color='blue')
                Vent1.calculate_VDP()
                window['-STATUS-'].update("VDP Calculated",text_color='green')
                window['-CALCVDP-'].update(button_color = 'green')
                updateImages()
                updateData()
            except:
                window['-STATUS-'].update("ERROR: VDP either couldnt run or be displayed for some reason...",text_color='red')
                continue

## --------------- CALCULATE CI Button ------------------- ##
        elif event == ('-CALCCI-'):
            try:
                window['-STATUS-'].update("Calculating CI...",text_color='blue')
                Vent1.calculate_CI()
                window['-STATUS-'].update("CI Calculated successfully",text_color='green')
                window['-CALCCI-'].update(button_color = 'green')
                updateImages()
                updateData()
            except:
                window['-STATUS-'].update("ERROR: CI couldnt run for some reason...",text_color='red')
                continue

## --------------- RUN TWIX Button ------------------- ##
        elif event == ('-RUNTWIX-'):
            pass
            # try:
            #     TWIX_path = values['TWIXpath']
            #     window['-STATUS-'].update("Processing TWIX file...",text_color='blue')
            #     Vent1.process_RAW(TWIX_path)
            #     window['-STATUS-'].update("TWIX Processed successfully",text_color='green')
            #     window['-RUNTWIX-'].update(button_color = 'green')
            #     TwixMontage = array3D_to_montage2D(Vent1.raw_HPvent)
            #     TwixMontageImage = arrayToImage(255*normalize(TwixMontage),(int(image_box_size*TwixMontage.shape[1]/TwixMontage.shape[0]),image_box_size))
            #     window['-TWIXIMAGE-'].update(data=TwixMontageImage)
            #     window['twixdate'].update(f'Twix Date: {Vent1.scanDateTime}')
            #     window['twixprotocol'].update(f'Twix Protocol: {Vent1.protocolName}')
            # except:
            #     window['-STATUS-'].update("ERROR: TWIX couldnt process for some reason...",text_color='red')
            #     continue

## --------------- CLEAR CACHE Button ------------------- ##
        elif event == ('-CLEARCACHE-'):
            try:
                del Vent1
            except:
                pass
            print('Clearing Cache...')
            window['notes'].update('')
            window['-INITIALIZE-'].update(button_color = 'lightgray')
            window['-CALCVDP-'].update(button_color = 'lightgray')
            window['-CALCCI-'].update(button_color = 'lightgray')
            window['-RUNTWIX-'].update(button_color = 'lightgray')
            window['-EXPORT-'].update(button_color = 'lightgray')
            window['genxeRadio'].update(False)
            window['genxeInputs'].update(visible=False)
            window['mepoRadio'].update(False)
            window['mepoInputs'].update(visible=False)
            window['clinicalRadio'].update(False)
            window['clinicalInputs'].update(visible=False)
            updateData()
            updateImages()
            window['-STATUS-'].update("Analysis Cache is cleared and ready for the next subject!...",text_color='blue')

            

## --------------- EXPORT Button ------------------- ##
        elif event == ('-EXPORT-'):
            '''Here we'll save 3 important things: the 4D data arrays, one of the DICOM datasets and one of the TWIX datasets,
            and all the many single variable inputs/outputs such as patient name, study ID, scan date/time, etc. To do this, we'll
            pickle everything in a single all-in-one file to be saved in the specified path and if desired in a static 'archive path, 
            and separately we'll save the arrays as Nifti's and full headers for TWIX as JSON files.'''

            # Did the user input their name??
            if values['userName'] == '':
                window['-STATUS-'].update("Don't forget to enter your Name or Initials at the very top right!...",text_color='red')
                continue

            # Did the user select an IRB??
            if not values['genxeRadio'] and not values['mepoRadio'] and not values['clinicalRadio']:
                window['-STATUS-'].update("Don't forget to select an IRB!...",text_color='red')
                continue

            # Create the EXPORT_path and fileName and populate class metadata dictionary with values from GUI input fields
            window['-STATUS-'].update("Exporting Data...",text_color='blue')
            today = date.today().strftime("%y%m%d")
            user = values['userName']
            targetPath = f'VentAnalysis_{user}_{today}/'
            EXPORT_path = os.path.join(values['exportpath'],targetPath)
            treatment = 'none'
            visit = '0'
            if values['genxeRadio']:
                fileName = f"Xe-{values['genxeID']}_{Vent1.metadata['StudyDate'][2:]}"
                if values['prealb']: fileName = f'{fileName}_preAlb';Vent1.metadata['treatment'] = 'preAlbuterol'
                elif values['postalb']: fileName = f'{fileName}_postAlb';Vent1.metadata['treatment'] = 'postAlbuterol'
                elif values['presil']: fileName = f'{fileName}_preSil';Vent1.metadata['treatment'] = 'preSildenafil'
                elif values['postsil']: fileName = f'{fileName}_postSil';Vent1.metadata['treatment'] = 'postSildenafil'
            elif values['mepoRadio']:
                fileName = f"Mepo{values['mepoID']}_{Vent1.metadata['StudyDate'][2:]}"
                if values['mepoVisit1']: fileName = f'{fileName}_visit1';Vent1.metadata['visit'] = 1
                elif values['mepoVisit2']: fileName = f'{fileName}_visit2';Vent1.metadata['visit'] = 2
                elif values['mepoVisit3']: fileName = f'{fileName}_visit3';Vent1.metadata['visit'] = 3
                if values['prealb_mepo']: fileName = f'{fileName}_preAlb';Vent1.metadata['treatment'] = 'preAlb'
                elif values['postalb_mepo']: fileName = f'{fileName}_postAlb';Vent1.metadata['treatment'] = 'postAlb'
            elif values['clinicalRadio']:
                fileName = f"Clinical_{values['clinicalID']}_{Vent1.metadata['StudyDate'][2:]}_visit{values['clinicalvisitnumber']}"
                if values['baseline']:
                    fileName = f'{fileName}_baseline';Vent1.metadata['treatment'] = 'none'
                    series_description = f"Vent_defect_VDP={np.round(Vent1.metadata['VDP'],1)}"
                    Vent1.SeriesDescription = series_description
                elif values['albuterol']:
                    fileName = f'{fileName}_Albuterol';Vent1.metadata['treatment'] = 'Albuterol'
                    series_description = f"VentBD_defect_VDP={np.round(Vent1.metadata['VDP'],1)}"
                    Vent1.SeriesDescription = series_description
            print(f'-- FileName: {fileName} --')
            print(f'-- FilePath: {EXPORT_path} --')
            if not os.path.isdir(EXPORT_path):
                os.makedirs(EXPORT_path)
            try:
                Vent1.metadata['analysisUser'] = user
                Vent1.metadata['fileName'] = fileName
                Vent1.metadata['DE'] = values['DE']
                Vent1.metadata['FEV1'] = values['FEV1']
                Vent1.metadata['FVC'] = values['FVC']
                Vent1.metadata['IRB'] = IRB
                Vent1.metadata['notes'] = values['notes']
            except:
                window['-STATUS-'].update("Could not add GUI metadata values to Class metadata...",text_color='red')
                print('\033[31mError adding GUI data to class metadata...\033[37m')

            #Export Nifti Arrays, DICOM header json, Class pickle, and screenshot
            Vent1.exportNumpys(EXPORT_path)
            Vent1.exportNifti(EXPORT_path,fileName)
            Vent1.dicom_to_json(Vent1.ds, json_path=os.path.join(EXPORT_path,f'{fileName}.json'))
            Vent1.pickleMe(pickle_path=os.path.join(EXPORT_path,f'{fileName}.pkl'))
            Vent1.dicom_template_path = values['dicom_template_path'].replace('"','')
            print(f"GUI: {Vent1.dicom_template_path}")
            Vent1.screenShot(path=os.path.join(EXPORT_path,f'{fileName}.png'))
            def get_slice_locations_from_folder(folder_path):
                slice_locations = []
                for filename in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, filename)
                    try:
                        ds = dicom.dcmread(file_path, stop_before_pixels=True)
                        slice_location = getattr(ds, 'SliceLocation', None)
                        slice_locations.append(float(slice_location))
                    except Exception as e:
                        print(f"Skipping file {filename}: {e}")
                        continue
                return slice_locations
            SlicLocs = get_slice_locations_from_folder(os.path.dirname(Vent1.dicom_template_path))
            print(SlicLocs)
            Vent1.exportDICOM(dicom_template_path=Vent1.dicom_template_path,
                              save_dir=f"{EXPORT_path}/defectDICOMS/",
                              SlicLocs=SlicLocs,
                              series_description=series_description)
            window['-STATUS-'].update("Data Successfully Exported...",text_color='green')

            if values['-ARCHIVE-'] == True:
                if os.path.isdir(ARCHIVE_path):
                    Vent1.pickleMe(pickle_path=os.path.join(ARCHIVE_path,f'{fileName}.pkl'))
                    window['-STATUS-'].update("Data Successfully Exported and Archived...",text_color='green')
                else:
                    window['-STATUS-'].update("Data Successfully Exported but not Archived...",text_color='orange')
                    print("Cant Archive because the path doesn't exist...")


            


'''Things to add (updated 3/27/2024):
 - CI colormap output in screenshot and GUI
 - Multiple VDPs calculated (linear binning, k-means) (LB done, Kmeans done, Adaptive Kmeans done)
 - show histogram?
 - edit mask
 - automatic segmentation using proton (maybe DL this?)
 - Denoise Option
 '''