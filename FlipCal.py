import os
import numpy as np
import matplotlib.pyplot as plt
import mapvbvd # ---------------------------------- for reading Twix files - __init__()
import pickle # ----------------------------------- for pickling class attributes - pickleMe() and unpickleMe()
from scipy.optimize import curve_fit # ------------ for fitting the Gas FID and decay - fit_Gas_FID() and getFlipAngle()
from scipy.optimize import differential_evolution # for fitting the DP FID - fit_DP_FID()
import concurrent.futures # ----------------------- for faster wiggles processing - fit_all_DP_FIDs()
import time # ------------------------------------- for calculating process times - fit_all_DP_FIDs()
from tqdm import tqdm # --------------------------- for console build - fit_all_DP_FIDs()
import datetime # --------------------------------- for analysis timestamps and dicom creation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # GUI stuff - printout()
import ismrmrd # ---------------------------------- For importing/exporting ISMRMRD formats - exportISMRMRD() and parseISMRMRD()
import json # ------------------------------------- For exporting Twix Header
import xml.etree.ElementTree as ET # -------------- To parse ISMRMRD header - parseISMRMRD()
import pydicom # ---------------------------------- For saving plots as dicoms - 
from pydicom.dataset import Dataset

class FlipCal:
    '''This class inputs raw FlipCal data (either as twix, matlab, or previously saved pickle),
    And performs all operations to determine our favorite info (gas decay, flip angle, RBC2MEM, etc.).
    INPUTS:
        Either a TWIX object or path, an ISMRMRD file (with consotrium-specified fields populated), 
        a .mat file from processing the data at the console, or a FlipCal Pickle file.
    ATTRIBUTES: listed below with comments for each one.
    METHODS (that do actual science): 
        SVD(): Performs SVD on the DP data array, and GAS data array and creates attributes
        fit_GAS_FID(): Fits a single gas RO to a decaying exponential.
        getFlipAngle(): Fits the gas decay (from SVD) and returns flip angle and new reference Voltage
        fit_DP_FID(): Fits a single DP RO to the 3-resonance-decay model. Returns a 3x5 matrix of parameters.
        fit_all_DP_FIDs(): Fits every DP RO and returns a 3x5x499 array of fit parameters (for RBCoscillations)
        calcualteWiggles(): get Wiggle amplitude
        printout(): creates a pretty figure of all data/analyses
    HELPER METHODS:
        parseTwix(): Given a twix, pulls the most important info and data into class attributes
        parseMatlab(): Given a .mat file, pulls the most important info and data into class attributes
        parseISMRMRD(): Given a path to ISMRMRD.h5 file, it will extract and populate class attributes
        exportISMRMRD(): Creates an ISMRMRD h5 file populated with XeCTC attributes
        pickleMe(): Pickles the class by building a dictionary of the class attributes.
        unpickleMe(): Unpickles a dictionary back into a class structure.
        extract_attributes(): converts attribute dicionary into a dictionary
        __repr__(): does the repr thing..'''
    def __init__(self,
                 twix_path =      None,
                 twix_object =    None,
                 pickle_dict =    None,
                 pickle_path =    None,
                 matlab_object =  None,
                 matlab_path =    None,
                 ismrmrd_path =   None):
        self.version = '250208_calibration' 
        # -- 241230, indication now a patientInfo key
        # -- 241206, RBC2MEM corrected based on excitation profile
        # -- 241115, Now saves plots as dicoms
        # -- 241030, ISMRMRD format now included
        # -- 241025, metadata updated to scanParameters and patientInfo
        # -- 241009, version numbers should contain the sequence type (cal here)
        # -- 240926, repr fixed, concurrent futures working?
        # -- 240925, amp is 2x hilber modulus
        # -- 240924, Fixed TE90 issue and twix_dict is now just header so it can be pickled
        # -- 240918, First full version
        self.processDate = datetime.date.today().strftime("%y%m%d")
        self.sequence = 'Calibration'
        # -- input data to be cast to attributes -- ##
        self.twix = '' # ---------- Note that the twix object itself cannot be pickled, but the entire header can
        self.matlab = ''
        self.FID = '' # ----------- Full FlipCal matrix
        self.t = '' # ------------- The time vector for a RO (dwellTime*(0:511))
        self.patientInfo = {'PatientName': '', # -- Most of the metadata is pulled in one of the parse functions
                         'PatientID': '',
                         'PatientAge': '',
                         'PatientSex': '',
                         'PatientDOB': '',
                         'PatientWeight': '',
                         'IRB': '',
                         'FEV1': '',
                         'FVC': '',
                         'LCI': '',
                         '6MWT': '',
                         'DE': '',
                         'indication': '',
                         '129XeEnrichment': 0.86}
        self.scanParameters = {
                         'ProtocolName': '',
                         'systemVendor': '',
                         'institutionName': '',
                         'B0fieldStrength': '',
                         'FlipAngle': '',
                         'DisFrequencyOffset': '',
                         'referenceAmplitude': '',
                         'TE': '',
                         'TR': '',
                         'GasFrequency': '',
                         'nFIDs': '',
                         'nPts': '',
                         'scanDate': '',
                         'scanTime': '',
                         'referenceVoltage': '',
                         'dwellTime': '',
                         'FieldStrength': '',
                         'FOV': '',
                         'nSkip': 100}
        
        # -- Created Attributes in SVD() -- ##
        self.singular_values_to_keep = 2
        self.noise = '' # --------- Single noise FID (column 0 of FID) 
        self.DP = '' # ------------ Dissolved phase FIDs (columns 1:499) 
        self.GAS = '' # ----------- Gas FIDs (columns 500:519) 
        self.DPfid = '' # --------- First U column in DP SVD (best representation of a single DP fid)
        self.DPdecay = '' # ------- First V column in DP SVD (captures decay due to depolarization)
        self.RBCosc = '' # -------- Second V column in Dp SVD (mostly captures wiggles)
        self.GASfid = '' # -------- First U column in GAS SVD (best representation of single GAS fid)
        self.gasDecay = '' # ------ First V column in GAS SVD (captures decay due to depolarization)
        # -- Attributes Created in fit_GAS_FID()
        self.gas_fit_params = '' # ---- A (1,5) of gas FID fit parameters [area, freq, phase, fwhmL, fwhmG]
        self.newGasFrequency = '' # --- Actual 129Xe Gas NMR frequency
        # -- Attributes Created in getFlipAngle()
        self.flipAngleFitParams = '' # -- parameters of gas Decay fit
        self.flip_angle = '' # ---------- Actual Delivered Flip Angle
        self.flip_err = '' # ------------ Error in calculated flip angle
        self.newVoltage = '' # ---------- Voltage that /would/ yield a 20° flip (to be given to DP)
        # -- Attributes Created in fit_DP_FID()
        self.DP_fit_params = '' #-- A (3,5) of DP FID fit parameters [row 0: RBC, row 1: MEM, row 2: GAS]
        # -- Attributes Created in fit_all_DP_FIDs()
        self.RO_fit_params = '' #---------- (3x5xN) array of DP fits (N=499 for our data typically)
        self.RBC2MEMsig = '' # ------------ vector of RBC/membrane signal ratios
        self.Mrbc = '' # ------------------ vector of RBC magnetizations (signal corrected for off-res excitation)
        self.Mmem = '' # ------------------ vector of MEM magnetizations (signal corrected for off-res excitation)
        self.RBC2MEMmag = '' # ------------ vector of RBC/membrane magnetization ratios (signal ratios corrected for off-resonance excitation)
        # -- Attributes Created in process()
        self.TE90 = '' # ---------- Time to 90° difference b/t RBC and MEM
        self.RBC2MEMsig = '' #  RBC2membrane signal ratio (fit of DP svd spectrum)
        self.RBC2MEMmag = '' #  RBC2membrane magnetization ration (signal ratio corrected by excitation angles)
        self.RBC2MEMdix = '' #  RBC2membrane ratio which the Dixon acquisition should use (assuming RBC resonance excitation)
        self.RBC2MEMsig_wiggles = '' #  Same as above but averaged from the dynamic RBC/mem calculation
        self.RBC2MEMmag_wiggles = '' #  Same as above but averaged from the dynamic RBC/mem calculation
        self.RBC2MEMdix_wiggles = '' #  Same as above but averaged from the dynamic RBC/mem calculation
        self.RBCppm = '' # -------- Chem shift of RBC peak
        self.MEMppm = '' # -------- Chem shift of MEM peak
        self.RBC2MEMmag_amp = '' # -The magnetization wiggle amplitude 
        ## -- Was a pickle or a pickle path provided? -- ##
        if pickle_path is not None:
            #print(f'\n \033[35m # ------ Pickle path provided: {pickle_path}. ------ #\033[37m')
            try:
                with open(pickle_path, 'rb') as file:
                    pickle_dict = pickle.load(file)
            except:
                print('\033[31mOpening Pickle from path and building arrays failed...\033[37m')
        
        if pickle_dict is not None:
            self.unPickleMe(pickle_dict)
            print(f"\n \033[35m # ------ FlipCal pickle {self.patientInfo['PatientName']} from {self.scanParameters['scanDate']} of shape {self.FID.shape} was loaded ------ #\033[37m")
        
        ## -- Was a twix object or path provided?
        if twix_path is not None:
            self.twix = mapvbvd.mapVBVD(twix_path)
            self.parseTwix()
            print(f"\n \033[35m# ------ FlipCal Twix Path {self.patientInfo['PatientName']} from {self.scanParameters['scanDate']} of shape {self.FID.shape} was loaded ------ #\033[37m")
        
        if twix_object is not None:
            self.twix = twix_object      
            self.parseTwix()
            print(f"\n \033[35m# ------ FlipCal Twix object {self.patientInfo['PatientName']} from {self.scanParameters['scanDate']} of shape {self.FID.shape} was loaded ------ #\033[37m")
        
        ## -- Was a matlab object or path provided?
        if matlab_object is not None:
            self.matlab = matlab_object
            self.parseMatlab()
            print(f"\n \033[35m# ------ FlipCal MatLAb object {self.patientInfo['PatientName']} from {self.scanParameters['scanDate']} of shape {self.FID.shape} was loaded ------ #\033[37m")
        
        if matlab_path is not None:
            import scipy.io
            self.matlab = scipy.io.loadmat(matlab_path)
            self.parseMatlab()
            print(f"\n \033[35m # ------ FlipCal MatLab path {self.patientInfo['PatientName']} from {self.scanParameters['scanDate']} of shape {self.FID.shape} was loaded ------ #\033[37m")
        
        ## -- Was a twix object or path provided?
        if ismrmrd_path is not None:
            self.parseISMRMRD(ismrmrd_path=ismrmrd_path)
            print(f"\n \033[35m # ------ FlipCal ISMRMRD path {self.patientInfo['PatientName']} from {self.scanParameters['scanDate']} of shape {self.FID.shape} was loaded ------ #\033[37m")
    
    def process(self):
        '''This does the entire Calibration processing pipeline /except/ the wiggles'''
        self.SVD()
        self.RMSnoise = np.std(np.concatenate((self.noise.real,self.noise.imag)))
        self.gas_fit_params, self.newGasFrequency = self.fit_GAS_FID()
        self.getFlipAngle()
        self.DP_fit_params = self.fit_DP_FID()
        self.RBC2MEMsig = self.DP_fit_params[0,0]/self.DP_fit_params[1,0]
        _,_,self.RBC2MEMmag,self.RBC2MEMdix = self.correctRBC2MEM(self.DP_fit_params[0,0],self.DP_fit_params[1,0],self.DP_fit_params[0,1],self.DP_fit_params[1,1])
        deltaPhase = (self.DP_fit_params[0,2] - self.DP_fit_params[1,2])
        deltaPhase = np.mod(np.abs(deltaPhase),180)
        deltaF = abs(self.DP_fit_params[0,1] - self.DP_fit_params[1,1])
        self.TE90 = int(self.scanParameters['TE'])*1e-3 + 1e3*(90 - deltaPhase)/(360 * deltaF) # -- in ms
        self.RBCppm = 1e6*(self.DP_fit_params[0,1]-self.DP_fit_params[2,1])/self.newGasFrequency
        self.MEMppm = 1e6*(self.DP_fit_params[1,1]-self.DP_fit_params[2,1])/self.newGasFrequency
        try:
            self.RBC2MEMsig_wiggles = self.RO_fit_params[0,0,:]/self.RO_fit_params[1,0,:]
            _,_,self.RBC2MEMmag_wiggles,self.RBC2MEMdix_wiggles = self.correctRBC2MEM(self.RO_fit_params[0,0,:],self.RO_fit_params[1,0,:],self.RO_fit_params[0,1,:],self.RO_fit_params[1,1,:]) #(Srbc,Smem,wrbc,wmem)
            self.RBC2MEMmag_amp = self.calcWiggleAmp(self.RBC2MEMmag_wiggles[100:])
            print(f"\033[33mThe RBC/MEM Signal ratio was {self.RBC2MEMsig} from SVD and {np.mean(self.RBC2MEMsig_wiggles[100:])} from wiggles\n\033[37m")
            print(f"\033[33mThe RBC/MEM Magnet ratio was {self.RBC2MEMmag} from SVD and {np.mean(self.RBC2MEMmag_wiggles[100:])} from wiggles\n\033[37m")
            print(f"\033[33mThe RBC/MEM Dixon  should be {self.RBC2MEMdix} from SVD and {np.mean(self.RBC2MEMdix_wiggles[100:])} from wiggles\n\033[37m")
        except:
            print('RBC2MEM not not in attributes. Need to run fit_all_DP_FIDs() method to get wiggles.')
            print(f"\033[33mThe RBC/MEM Signal ratio was {self.RBC2MEMsig} from SVD \n\033[37m")
            print(f"\033[33mThe RBC/MEM Magnet ratio was {self.RBC2MEMmag} from SVD \n\033[37m")
            print(f"\033[33mThe RBC/MEM Dixon  should be {self.RBC2MEMdix} from SVD\n\033[37m")
        self.processDate = datetime.date.today().strftime("%y%m%d")
    
    def parseTwix(self):
        '''Fetches some of our favorite parameters from the twix object header.
        Most of these are stored in the scanParameters dictionary'''
        # -- INPUT ATTRIBUTES -- ##
        self.twix.image.squeeze = True # --- remove singleton dimensions of data array
        self.twix.image.removeOS = False # - Keep all 512 datapoints (don't reduce to 256)
        self.FID = self.twix.image[''] # --- the FID data array (index 0 = noise, 1:499 = DP, 500:519 = GAS)
        self.calibration_dict = self.extract_attributes(self.twix.hdr) ## -- converts twix header to dictionary
        self.patientInfo['PatientName'] = self.twix.hdr.Config['PatientName']
        self.patientInfo['PatientDOB'] = self.twix.hdr.Config['PatientBirthDay']
        self.patientInfo['PatientSex'] = self.twix.hdr.Config['PatientSex']
        self.patientInfo['PatientWeight'] = self.twix.hdr.Dicom['flUsedPatientWeight']
        self.patientInfo['PatientAge'] = self.twix.hdr.Dicom['flPatientAge']
        self.scanParameters['scanDateTime'] = self.twix.hdr.Config['PrepareTimestamp']
        self.scanParameters['scanDate'], self.scanParameters['scanTime'] = self.twix.hdr.Config['PrepareTimestamp'].split()
        self.scanParameters['systemVendor'] = self.twix.hdr.Dicom.Manufacturer
        self.scanParameters['scannerSoftwareVersion'] = self.twix.hdr.Dicom.SoftwareVersions
        self.scanParameters['institutionName'] = self.twix.hdr.Dicom.InstitutionName
        self.scanParameters['B0fieldStrength'] = self.twix.hdr.Meas['flNominalB0']
        self.scanParameters['FlipAngle'] = float(self.twix.hdr.Meas["adFlipAngleDegree"].split(" ")[0])
        self.scanParameters['FlipAngle_DP'] = float(self.twix.hdr.Meas["adFlipAngleDegree"].split(" ")[1])
        self.scanParameters['XenonLarmorFrequency'] = float(self.twix.hdr.Meas["alLarmorConstant"].split(" ")[0])
        self.scanParameters['ProtocolName'] = self.twix.hdr.Config['SequenceDescription']# Also in Meas
        self.scanParameters['referenceAmplitude'] = self.twix.hdr.Dicom['flTransRefAmpl'] #Also in Meas and Protocol
        self.scanParameters['FOV'] = float(self.twix.hdr.Config.ReadFoV) #Field of View (I guess needed for ISMRMRD on calibrations??)
        self.scanParameters['TE'] = self.twix.hdr.Meas['alTE'].split(' ')[0] #Also in Protocol
        self.scanParameters['TR'] = self.twix.hdr.Config['TR'].split(' ')[0]
        self.scanParameters['GasFrequency'] = self.twix.hdr.Dicom['lFrequency'] # -- Gas Frequency in Hz
        self.scanParameters['DisFrequencyOffset'] = self.twix.hdr.Phoenix["sWipMemBlock", "alFree", "4"]
        self.scanParameters['nFIDs'] = self.twix.hdr.Config['NRepMeas']
        self.scanParameters['nPts'] = self.twix.hdr.Config['VectorSize'] # This assumes oversampling
        self.scanParameters['scanDate'] = self.twix.hdr.MeasYaps[("tReferenceImage0",)].strip('"').split(".")[-1][:8] # - scanDate
        self.scanParameters['scanTime'] = self.twix.hdr.Config['PrepareTimestamp'][-8:] # -- scan time
        self.scanParameters['referenceVoltage'] = self.twix.hdr.MeasYaps[('sWipMemBlock', 'alFree', '0')] # -- reference voltage
        self.scanParameters['dwellTime'] = self.twix.hdr.Config['DwellTime']*1e-9 # in seconds
        self.scanParameters['FieldStrength'] = self.twix.hdr.Dicom['flMagneticFieldStrength'] # - B0 strength
        self.scanParameters['PulseDuration'] = float(self.twix.hdr.Meas['alTD'].split(' ')[0]) # - Pulse Duration in us
        self.scanParameters['dissolvedFrequencyOffset'] = self.twix.hdr.MeasYaps[('sWipMemBlock', 'alFree', '4')]
        self.t = np.arange(self.FID.shape[0])*self.scanParameters['dwellTime']
    
    def parseMatlab(self):
        '''Note that the newest version of the matlab Processing code (8/2024 and later) contains
        the twix object from mapVBVD, so maybe just use that to populat the header??'''
        self.FID = self.matlab['theFID']
        self.patientInfo['PatientName'] = self.matlab['twix_obj']['hdr'][0][0]['Config'][0][0]['PatientName'][0][0][0]
        self.scanParameters['ProtocolName'] = self.matlab['twix_obj']['hdr'][0][0]['Config'][0][0]['SequenceDescription'][0][0][0]
        self.scanParameters['PatientWeight'] = self.matlab['twix_obj']['hdr'][0][0]['Dicom'][0][0]['flUsedPatientWeight'][0][0][0][0]
        self.scanParameters['referenceAmplitude'] = self.matlab['twix_obj']['hdr'][0][0]['Dicom'][0][0]['flTransRefAmpl'][0][0][0][0]
        self.scanParameters['TE'] = self.matlab['te'][0][0]
        self.scanParameters['TR'] = self.matlab['tr'][0][0]
        self.scanParameters['GasFrequency'] = self.matlab['freq'][0][0]
        self.scanParameters['nFIDs'] = self.matlab['twix_obj']['hdr'][0][0]['Config'][0][0]['NRepMeas'][0][0][0][0]
        self.scanParameters['nPts'] = self.matlab['nPts'][0][0]
        #self.scanParameters['scanDateTime'] = self.matlab['twix_obj']['hdr'][0][0]['Config'][0][0]['PrepareTimeStamp'][0][0][0][0]
        self.scanParameters['referenceVoltage'] = self.matlab['VRef'][0][0]
        self.scanParameters['dwellTime'] = self.matlab['dwell_time'][0][0]
        self.t = np.arange(self.FID.shape[0])*self.scanParameters['dwellTime']
        self.RO_fit_params = self.matlab['ROfitParams']
        self.RO_fit_params[:,0,:] = self.matlab['ROfitParams'][:,1,:]
        self.RO_fit_params[:,1,:] = self.matlab['ROfitParams'][:,0,:]
        self.RBC2MEM = self.RO_fit_params[0,0,:]/self.RO_fit_params[1,0,:]
        self.RBC2MEMavg = np.mean(self.RBC2MEM[self.scanParameters['nSkip']:])
        self.flipAngleFitParams = self.matlab['fitparams'][0]
        self.flip_angle = self.flipAngleFitParams[1]*180/np.pi
        self.flip_err = 0
        self.scanParameters['newVoltage'] = self.matlab['VRefScaleFactor']*self.matlab['VRef']
        self.GASfid = self.FID[:,500]
        self.DPfid = self.matlab['disData_avg']
        [self.Ugas,self.Sgas,self.VTgas] = np.linalg.svd(self.FID[:,500:])
        self.gasDecay = np.abs(self.VTgas[0,:]*self.Sgas[0])
        def hilbert(S):
            F = np.fft.fftshift(np.fft.fft(S))
            F[:int(len(S)/2 + 1)] = 0
            H = np.fft.ifft(np.fft.fftshift(F))
            return H
        hilb = hilbert(self.RBC2MEM[self.scanParameters['nSkip']:])
        self.RBC2MEMamp = np.mean(np.abs(hilb))
        try:
            self.DP_fit_params = self.matlab['disFit']
            self.gas_fit_params = self.matlab['gasFit'][0]
            self.newGasFrequency = self.scanParameters['GasFrequency'] + self.gas_fit_params[1]
        except:
            print('Older version of matlab file was parsed. need to run process to get DP and GAS fit params...')
    
    def parseISMRMRD(self,ismrmrd_path):
        """Function to read an ISMRMRD .h5 file and reconstruct the header_dict and FID data."""
        ismrmrd_object = ismrmrd.Dataset(ismrmrd_path,'/dataset',create_if_needed=False) # gets IRMRMRD object
        print('Pulling ISMRMRD Header...')
        header_str = ismrmrd_object.read_xml_header().decode('utf-8') #convert ismrmrd object header to xml then to bytestring
        root = ET.fromstring(header_str)
        def parse_ismrmrd_header(root):
            def xml_to_dict(elem):
                if len(elem) == 0:  # No children
                    return elem.text
                else:
                    return {child.tag.split('}')[-1]: xml_to_dict(child) for child in elem}  # Remove namespace if present
            header_dict = {root.tag.split('}')[-1]: xml_to_dict(root)}  # Remove namespace from root tag
            return header_dict
        self.calibration_dict = parse_ismrmrd_header(root) # from xml bytestring make dictionary
        # (editor's note: this is stupidly complicated and requires packages outside ismrmrd)
        self.scanParameters['scanDate'] = self.calibration_dict['ismrmrdHeader']['studyInformation']['studyDate']
        self.patientInfo['PatientName'] = self.calibration_dict['ismrmrdHeader']['subjectInformation']['patientID']
        self.scanParameters['systemVendor'] = self.calibration_dict['ismrmrdHeader']['acquisitionSystemInformation']['systemVendor']
        self.scanParameters['institutionName'] = self.calibration_dict['ismrmrdHeader']['acquisitionSystemInformation']['institutionName']
        self.scanParameters['B0fieldStrength'] = float(self.calibration_dict['ismrmrdHeader']['acquisitionSystemInformation']['systemFieldStrength_T'])
        self.scanParameters['TE'] = int(self.calibration_dict['ismrmrdHeader']['sequenceParameters']['TE'])
        self.scanParameters['TR'] = int(self.calibration_dict['ismrmrdHeader']['sequenceParameters']['TR'])
        self.scanParameters['FlipAngle'] = float(self.calibration_dict['ismrmrdHeader']['sequenceParameters']['flipAngle_deg'])
        self.scanParameters['FOV'] = self.calibration_dict['ismrmrdHeader']['encoding']['reconSpace']['fieldOfView_mm']['x']
        ## -- Pulling userParameters here
        ns = {'ns': 'http://www.ismrm.org/ISMRMRD'}
        for user_param in root.findall(".//ns:userParameters/ns:userParameterLong", namespaces=ns):
            param_name = user_param.find("ns:name", namespaces=ns).text
            param_value = user_param.find("ns:value", namespaces=ns).text
            if param_name == 'xe_center_frequency':
                self.scanParameters['GasFrequency'] = float(param_value)
            elif param_name == 'xe_dissolved_offset_frequency':
                self.scanParameters['DisFrequencyOffset'] = float(param_value)
            else:
                self.scanParameters[param_name] = param_value
                print('During ISMRMRD Parse I found {param_name} with value {param_value}')
        print(f'Pulling {ismrmrd_object.number_of_acquisitions()} acquisitions from ISMRMRD data...')
        acquisitions = []
        for i in range(ismrmrd_object.number_of_acquisitions()):
            acq = ismrmrd_object.read_acquisition(i)
            acquisitions.append(np.squeeze(acq.data))
        self.FID = np.stack(acquisitions, axis=-1)
        acquisition_header = ismrmrd_object.read_acquisition(0).getHead()
        self.scanParameters['dwellTime'] = acquisition_header.sample_time_us*1e-6
        self.t = np.arange(self.FID.shape[0])*self.scanParameters['dwellTime']
        ismrmrd_object.close()
    
    def SVD(self):
        """Performs Singular Value Decomposition for analysis"""
        def flipCheck(a,b): # -- SVD can randomize the sign of data, lets check and fix if needed
            if np.sign(a.real[0]) != np.sign(b.real[0]):
                a = -a
            return a
        ## -- Separate NOISE, DP, and GAS arrays -- ##
        self.noise = self.FID[:,0] # --------------------------- The noise RO
        self.DP = self.FID[:,1:500] # -------------------------- All DP ROs
        self.GAS = self.FID[:,500:] # -------------------------- All GAS ROs
        ## -- DP -- Attributes are created for first U column (best single DP RO), first V column (decay), And second V column (oscillations)##
        [U,S,VT] = np.linalg.svd(self.DP[:,self.scanParameters['nSkip']:])
        self.DPfid = flipCheck(U[:,0]*S[0]**2,self.DP[:,0]) # --------------- The best representation of a single DP readout
        self.DPdecay = VT[0,:]*S[0] # --------------------------------------- The DP signal decay across readouts
        self.RBCosc = VT[1,100:]*S[1] # ------------------------------------- The RBC oscillations
        # S[self.singular_values_to_keep:] = 0
        # Smat = np.zeros((self.DP.shape[0],self.DP.shape[1]))
        # Smat[:len(S),:len(S)] = np.diag(S)
        # self.smoothDP = U @ Smat  @ VT #our experience shows this SVDing to denoise doesn't really help much
        ## -- GAS -- Attributes are created for first U column (single RO), and first V column (gas signal decay) ## 
        [U,S,VT] = np.linalg.svd(self.GAS)
        self.GASfid = flipCheck(U[:,0]*S[0]**2,self.GAS[:,0]) # ------------ The best representation of a single Gas readout
        self.gasDecay = np.abs(VT[0,:]*S[0]) # - The gas signal decay across readouts
    
    def FIDFitfunction(self, t, A, f, phi, L, G):
        '''t: time [s], A: amplitude [arb], f: frequency [Hz], phi: phase [°], L: Lorentzian fwhm [Hz], G: Gaussian fwhm [Hz]'''
        return A * np.exp(1j*phi*np.pi/180) * np.exp(1j * f * 2 * np.pi * t) * np.exp(-t * np.pi * L) * np.exp(-t**2 * 2* np.log(2) * G**2)
    
    def fit_GAS_FID(self,FID=None):
        '''Fits the SVD gas RO. t [sec], A [arb], phi [radians], f [Hz], L [Hz]. Note L * pi = 1/T2star'''
        print('\033[32mFitting Gas FID...\033[37m')
        if FID is None: # -- If no FID is input, it uses the SVD Gas FID
            FID = self.GASfid
        def gasFitFunction(t, A, f, phi, L, G):
            x = A * np.exp(1j*phi * np.pi/180) * np.exp(1j * f * 2 * np.pi * t) * np.exp(-t * np.pi * L) * np.exp(-t**2 * 2* np.log(2) * G**2)
            return np.concatenate((x.real,x.imag))
        data = np.concatenate((FID.real,FID.imag))
        gas_fit_params, _ = curve_fit(gasFitFunction, self.t, data, p0=[np.max(np.abs(FID)), 0, 0, 10, 40])
        [A, f, phi, L, G] = gas_fit_params
        newGasFrequency = self.scanParameters['GasFrequency'] + f
        print(f"\033[36mGasFID: Area -- Frequency -- Phase -- FWHML -- FWHMG\033[37m")
        print(f"\033[36m        \033[37m {np.round(A,3)} -- {np.round(f,0)} -- {np.round(phi,1)} -- {np.round(L,1)} -- {np.round(G,1)}")
        print(f"\033[34mGas FID is at frequency \033[32m{newGasFrequency}\033[34m Hz with L = {L} Hz, G = {G} Hz, ϕ = {phi*180/np.pi}°\033[37m")
        return gas_fit_params, newGasFrequency
    
    def getFlipAngle(self,Decay=None):
        '''Fits the Gas Decay in order to find flip angle and, in turn, the voltage correction.'''
        if Decay is None: # -- If no Decay is input, it uses the SVD Decay from the Gas dtata
            Decay = self.gasDecay
        gasDecay_fit_function = lambda x, a, b, c: a * np.cos(b) ** (x - 1) + c
        guess = [np.max(Decay), 20 * np.pi / 180, 0]  # max, 20 degrees in radians, and 0
        excitation = np.arange(1, len(Decay) + 1)
        self.flipAngleFitParams, pcov = curve_fit(gasDecay_fit_function, excitation, Decay, p0=guess)
        param_err = np.sqrt(np.diag(pcov))
        self.flip_angle = np.abs(self.flipAngleFitParams[1] * 180 / np.pi)
        self.flip_err = param_err[1] * 180 / np.pi
        print(f'\033[36mI calculated a flip angle = {np.round(self.flip_angle,1)} ± {np.round(self.flip_err,1)}\033[37m')
        try:
            self.newVoltage = self.scanParameters['referenceVoltage']*self.scanParameters['FlipAngle']/self.flip_angle
            print(f"\033[36m...So you should change your reference voltage from \033[32m{np.round(self.scanParameters['referenceVoltage'],1)} \033[36mto \033[32m{np.round(self.newVoltage,1)}\033[37m")
        except:
            print(f"\033[36m---Either referenceVoltage or FlipAngle is not in scanParameters\033[37m")
    
    def fit_DP_FID(self,FID=None,printResult = True):
        '''Give it a DP FID, and it will fit to the 3-resonance-decay model and 
        return the fitted 15 parameters in a 3x5 array.'''
        if FID is None: # -- If no FID is input, it uses the SVD DPfid
            FID = self.DPfid
        t = np.arange(len(self.DPfid)) * self.scanParameters['dwellTime']
        S = np.concatenate((FID.real,FID.imag))
        def FIDfunc_cf(t, a,b,c,d,e,f,g,h,i,j,k,l,m,n,o):
            A = np.array([[a,b,c,d,e],[f,g,h,i,j],[k,l,m,n,o]])
            frbc = (A[0, 0] * np.exp(1j * np.pi / 180 * A[0, 2]) * 
                    np.exp(1j * 2 * np.pi * t * A[0, 1]) * 
                    np.exp(-t * np.pi * A[0, 3]) * 
                    np.exp(-t**2 * A[0, 4]**2 * 4 * np.log(2)))
            
            fmem = (A[1, 0] * np.exp(1j * np.pi / 180 * A[1, 2]) * 
                    np.exp(1j * 2 * np.pi * t * A[1, 1]) * 
                    np.exp(-t * np.pi * A[1, 3]) * 
                    np.exp(-t**2 * A[1, 4]**2 * 4 * np.log(2)))
            
            fgas = (A[2, 0] * np.exp(1j * np.pi / 180 * A[2, 2]) * 
                    np.exp(1j * 2 * np.pi * t * A[2, 1]) * 
                    np.exp(-t * np.pi * A[2, 3]) * 
                    np.exp(-t**2 * A[2, 4]**2 * 4 * np.log(2)))
            
            f = frbc + fmem + fgas
            return np.concatenate((f.real,f.imag))
        
        def residual_de(A,t,S):
            newS = FIDfunc_cf(t,*A)
            return np.sum(np.concatenate([S.real - newS.real, S.imag - newS.imag])**2)

        Gasfreq = self.scanParameters['GasFrequency'] # - Target Gas frequency from Twix Header
        DPfreq = self.scanParameters['dissolvedFrequencyOffset'] # - Target offset from Twix Header
        ppmOffset = DPfreq / (Gasfreq*1e-6)
        print(f'\033[36mThis experiment excited at \033[32m{DPfreq} Hz ({np.round(ppmOffset,0)} ppm)\033[36m higher than Gas.\033[37m')
        RBCexpectedHz = (218*Gasfreq*1e-6) - DPfreq
        MEMexpectedHz = (197*Gasfreq*1e-6) - DPfreq
        print(f'\033[36mso I expect RBC near \033[32m{np.round(RBCexpectedHz,1)} Hz\033[36m, MEM near \033[32m{np.round(MEMexpectedHz,1)} Hz\033[36m, and Gas near RBC at \033[32m{np.round(-DPfreq,1)} Hz\033[36m,\033[37m')
        lbounds = np.array([[0,RBCexpectedHz-358,-180,0,0],[0,MEMexpectedHz-358,-180,0,0],[0,-DPfreq-1000,-180,0,0]])
        ubounds = np.array([[1,RBCexpectedHz+358,180,1000,1],[1,MEMexpectedHz+358,180,1000,400],[1,-DPfreq+1000,180,100,1]])
        debounds = [(lbounds.flatten()[k],ubounds.flatten()[k]) for k in range(len(lbounds.flatten()))]
        for fit_iteration in range(5):
            diffev = differential_evolution(residual_de, bounds=debounds, args=(t, S), maxiter=30000, tol=1e-9, popsize = 3, mutation = (0.5,1.0), recombination=0.7)
            de = np.reshape(diffev.x,(3,5))
            #Check to see if we ran into a frequency bound on the fit - this usually indicates misfitting and will yield erroneous values. Refit if we did.
            if (de[0,1] > (RBCexpectedHz-357)) and (de[0,1] < (RBCexpectedHz+357)) and (de[1,1] > (MEMexpectedHz-357)) and (de[1,1] < (MEMexpectedHz+357)):
                break
            else:
                print(f"Refitting {fit_iteration}/5.  RBC:{de[0,1]}  MEM:{de[1,1]}")
        
        if(printResult):
            print(f"\033[36m     Area -- Frequency -- Phase -- FWHML -- FWHMG\033[37m")
            print(f"\033[36mRBC:\033[37m {np.round(de[0,0]/de[1,0],3)} -- {np.round(de[0,1],0)} -- {np.round(de[0,2],1)} -- {np.round(de[0,3],1)} -- {np.round(de[0,4],1)}")
            print(f"\033[36mMEM:\033[37m {np.round(de[1,0]/de[1,0],3)} -- {np.round(de[1,1],0)} -- {np.round(de[1,2],1)} -- {np.round(de[1,3],1)} -- {np.round(de[1,4],1)}")
            print(f"\033[36mGAS:\033[37m {np.round(de[2,0]/de[1,0],3)} -- {np.round(de[2,1],0)} -- {np.round(de[2,2],1)} -- {np.round(de[2,3],1)} -- {np.round(de[2,4],1)}")
            _,_,RBC2MEMmag,RBC2MEMdix = self.correctRBC2MEM(de[0,0],de[1,0],de[0,1],de[1,1])
            print(f"\033[36mRBC2MEM magnitude = {RBC2MEMmag},  RBC2MEM dixon = {RBC2MEMdix},")
        
        return de # ROWS are RBC [0], MEM [1], GAS [2].  COLS are Area[0], frequency[1] in Hz, phase[2] in °, L[3] in Hz, G[4] in Hz
    
    def fit_all_DP_FIDs(self,**kwargs):
        '''Fits every DP RO to the 3-resonance-decay model - This is where our RBC oscillations come from.
        OUTPUTS:
            RO_fit_params: This is a 3x5xN array (N is number of DP readouts, 499 for us)
            RBC2MEMsig: RBC/membrane ratio for all readouts as a vector
            Mrbc: RBC magnetization (RBC signal corrected by excitation power)
            Mmem: MEM magnetization (MEM signal corrected by excitation power)
            RBC2MEMmag: Ratio of RBC/membrane magnetization (this corrects for different excitation power at different frequencies)
        Takes awhile. Need to get this going faster somehow.'''
        if 'data' in kwargs:
            print('You gave me data')
            internalDataMarker = False
            data = kwargs['data']
        else:
            print('You did not give me data. Using self.DP')
            internalDataMarker = True
            data = self.DP
        goFast = kwargs.get('goFast', False)
        RO_fit_params = np.zeros((3, 5, data.shape[1]))
        start_time = time.time()

        #-- Fast. Uses all CPU cores to process the data faster (default). May slow up your computer for a bit though
        if(goFast):
            print("\033[35mFitting all DP FIDs using Concurrent Futures. This may take awhile...\033[37m")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                print('\n Generating futures list...')
                futures = [executor.submit(self.fit_DP_FID, self.FID[:, RO+1],False) for RO in tqdm(range(498))]
                print('Fitting Data...')
                concurrent.futures.as_completed(futures)
                RO = 0
                for future in tqdm(futures):
                    RO_fit_params[:, :, RO+1] = future.result()
                    RO += 1
        
        #-- Slow. Use this if you're not pressed for time and need your CPU freed up (about 20-30 min)
        if(not goFast):
            print("\033[35mFitting all DP FIDs. This may take awhile...\033[37m")
            for RO in tqdm(range(data.shape[1])):
            #for RO in tqdm(range(3)):
                RO_fit_params[:,:,RO] = self.fit_DP_FID(self.FID[:,RO],printResult=False)
        
        print(f"Time to fit all readouts: {np.round((time.time()-start_time)/60,2)} min")
        if internalDataMarker:
            print('casting fit results into attributes RO_fit_params, RBC2MEM, and RBC2MEMavg...')
            self.RO_fit_params = RO_fit_params
            self.RBC2MEMsig_wiggles = self.RO_fit_params[0,0,:]/self.RO_fit_params[1,0,:]
            _,_,self.RBC2MEMmag_wiggles,self.RBC2MEMdix_wiggles = self.correctRBC2MEM(self.RO_fit_params[0,0,:],self.RO_fit_params[1,0,:],self.RO_fit_params[0,1,:],self.RO_fit_params[1,1,:]) #(Srbc,Smem,wrbc,wmem)
            self.RBC2MEMmag = self.calcWiggleAmp(self.RBC2MEMmag_wiggles)
        else:
            print('Returning Values')
            self.results = RO_fit_params
            return RO_fit_params
    
    def kappa(self,w, T = None):
        '''You give me your pulse time T, and resonance-offset frequency w in Hz, and I'll
        tell you what the relative amplitude of B1 you got.'''
        if T is None:
            T = self.scanParameters['PulseDuration']*1e-6
        wrad = w*2*np.pi
        return (2 / T) * np.sin(wrad * T / 2) * (1/wrad - wrad / (wrad**2 - (2 * np.pi / T)**2))
    
    def correctRBC2MEM(self,Srbc,Smem,wrbc,wmem): 
        '''Given an rbc and mem signal and the offset frequencies of rbc and mem, returns the rbc/mem magnetizations and ratio
        essentially this corrects for the fact that spins experience different flip angles based on their frequency (see Bechtel MRM 2023)'''
        Mrbc = Srbc/np.sin(self.kappa(wrbc)*np.pi / 9)
        Mmem = Smem/np.sin(self.kappa(wmem)*np.pi / 9)
        RBC2MEMmag = Mrbc/Mmem
        RBC2MEMdix = RBC2MEMmag*np.sin(np.pi/9)/np.sin(np.pi/9*self.kappa(wmem-wrbc,T=710*1e-6))
        return Mrbc, Mmem, RBC2MEMmag, RBC2MEMdix

    def calcWiggleAmp(self,wiggles): 
        def hilbert(S):
            F = np.fft.fftshift(np.fft.fft(S))
            F[:int(len(S)/2 + 1)] = 0
            H = np.fft.ifft(np.fft.fftshift(F))
            return H
        try:
            hilb = hilbert(wiggles)
            return 2*np.mean(np.abs(hilb)) # times 2 for Pk-pk
        except:
            print('You gotta run fit_all_DP_FIDs() first...')
    
    ## ------ Plot Draw Functions ------ ##
    def draw_GAS_phasor(self,ax4):
        gasPhasor_nTE = self.FIDFitfunction(-435e-6,*self.gas_fit_params)
        gasPhasor_0 = self.FIDFitfunction(-435e-6,*self.gas_fit_params)
        ax4.set_title('Gas Phasor')
        ax4.plot([0,1],[0,1],c=(1,1,1))
        ax4.vlines(x = [0],ymin=[-1],ymax=[1], color = (0.3,0.5,0.8),linestyle='dashed')
        ax4.hlines(y = [0],xmin=[-1],xmax=[1], color = (0.3,0.5,0.8),linestyle='dashed')
        ax4.arrow(0,0,gasPhasor_nTE.real/abs(gasPhasor_nTE),gasPhasor_nTE.imag/abs(gasPhasor_nTE),head_width=0.05, head_length=0.1,linewidth=1,linestyle='dashed')
        ax4.arrow(0,0,gasPhasor_0.real/abs(gasPhasor_0),gasPhasor_0.imag/abs(gasPhasor_0),head_width=0.05, head_length=0.1,linewidth=2)
        ax4.set_xlim([-1,1])
        ax4.set_axis_off()
    
    def draw_GAS_FID(self,ax5):
        ax5.set_title('Gas FID')
        T = np.linspace(-0.43e-3,10e-3,1000)
        gasFit = self.FIDFitfunction(T,*self.gas_fit_params)
        ax5.plot(self.t*1000,self.GASfid.real,c=(0,0.7,0)) # -- Plot the actual Gas FID
        ax5.plot(self.t*1000,self.GASfid.imag,c=(0.4,0,0.7))
        ax5.plot(self.t*1000,abs(self.GASfid),c=(0,0,0))
        ax5.plot(T*1000,gasFit.real,linestyle='dashed',c=(0,0.7,0)) # -- Now plot the fit (dashed)
        ax5.plot(T*1000,gasFit.imag,linestyle='dashed',c=(0.4,0,0.7))
        ax5.plot(T*1000,abs(gasFit),linestyle='dashed',c=(0,0,0))
        ax5.plot(T*1000,-abs(gasFit),linestyle='dashed',c=(0,0,0))
        ax5.hlines(y=0,xmin=0,xmax=self.t[-1]*1000,linewidth=0.5,color=(0.3,0.3,0.3),linestyle='dashed')
        ax5.vlines(x=0,ymin=-0.6,ymax=0.6,linewidth=0.5,color=(0.3,0.3,0.3),linestyle='dashed')
        ax5.set_xlim([-0.43,10])
        ax5.set_ylim([-np.max(abs(self.GASfid)),np.max(abs(self.GASfid))])
        ax5.set_xlabel('time [ms]')
    
    def draw_GAS_spectrum(self,ax6):
        ax6.set_title('Gas Spectrum')
        w = np.linspace(-0.5,0.5,len(self.GASfid))/self.scanParameters['dwellTime']
        F = np.fft.fftshift(np.fft.fft(self.GASfid))
        ax6.plot(w,abs(F),c=(0,0,0))
        ax6.plot(w,F.real,c=(0,0.7,0))
        ax6.plot(w,F.imag,c=(0.4,0,0.7))
        ax6.set_xlim([-1500,2500])
        ax6.text(1000,0.9*np.max(abs(F)),f"Gas FID Fit",fontsize=10,color=(0,0,0.7))
        ax6.text(1000,0.7*np.max(abs(F)),f"  ɷ = {np.round(self.newGasFrequency,0)} ({np.round(self.newGasFrequency-self.scanParameters['GasFrequency'],0)}) [Hz]",fontsize=9)
        ax6.text(1000,0.5*np.max(abs(F)),f"  ɸ = {np.round(self.gas_fit_params[2]*180/np.pi,0)}°",fontsize=9)
        ax6.text(1000,0.3*np.max(abs(F)),f"  L = {np.round(self.gas_fit_params[3],0)} [Hz]",fontsize=9)
        ax6.text(1000,0.1*np.max(abs(F)),f"  G = {np.round(self.gas_fit_params[4],0)} [Hz]",fontsize=9)
        ax6.set_xlabel('frequency [Hz]')
    
    def draw_DP_phasor(self,axc):
        TE = int(self.scanParameters['TE'])*1e-6
        RBC_n0 = self.FIDFitfunction(0, *self.DP_fit_params[0,:])
        MEM_n0 = self.FIDFitfunction(0, *self.DP_fit_params[1,:])
        GAS_n0 = self.FIDFitfunction(0, *self.DP_fit_params[2,:])
        RBC_nTE = self.FIDFitfunction(-TE, *self.DP_fit_params[0,:])
        MEM_nTE = self.FIDFitfunction(-TE, *self.DP_fit_params[1,:])
        GAS_nTE = self.FIDFitfunction(-TE, *self.DP_fit_params[2,:])
        axc.set_title(f"DP _nTE (t=-{self.scanParameters['TE']})")
        axc.plot([0,1],[0,1],c=(1,1,1))
        axc.vlines(x = [0],ymin=[-1],ymax=[1], color = (0.3,0.5,0.8),linestyle='dashed')
        axc.hlines(y = [0],xmin=[-1],xmax=[1], color = (0.3,0.5,0.8),linestyle='dashed')
        axc.arrow(0,0,RBC_nTE.real/abs(MEM_nTE),RBC_nTE.imag/abs(MEM_nTE),color=(0.8,0,0),head_width=0.05, head_length=0.1,linewidth=2)
        axc.arrow(0,0,MEM_nTE.real/abs(MEM_nTE),MEM_nTE.imag/abs(MEM_nTE),color=(0,0.8,0),head_width=0.05, head_length=0.1,linewidth=2)
        axc.arrow(0,0,GAS_nTE.real/abs(MEM_nTE),GAS_nTE.imag/abs(MEM_nTE),color=(0,0,0.8),head_width=0.05, head_length=0.1,linewidth=2)
        axc.set_ylim([-1,1])
        axc.set_xlim([-1,1])
        axc.set_axis_off()
    
    def draw_DP_FID(self,ax8):
        T = np.linspace(-int(self.scanParameters['TE']),0.01,1000)
        RBC_t = self.FIDFitfunction(T,*self.DP_fit_params[0,:])
        MEM_t = self.FIDFitfunction(T,*self.DP_fit_params[1,:])
        GAS_t = self.FIDFitfunction(T,*self.DP_fit_params[2,:])
        TOT_t = RBC_t + MEM_t + GAS_t
        ax8.set_title('DP FID')
        ax8.hlines(y=0,xmin=0,xmax=self.t[-1]*1000,linewidth=0.5,color=(0.3,0.3,0.3),linestyle='dashed')
        ax8.vlines(x=0,ymin=-0.6,ymax=0.6,linewidth=0.5,color=(0.3,0.3,0.3),linestyle='dashed')
        ax8.plot(self.t*1000,self.DPfid.real,c=(0,0.7,0))
        ax8.plot(self.t*1000,self.DPfid.imag,c=(0.4,0,0.7))
        #ax8.plot(T*1000,abs(TOT_t),c=(0,0,0),linestyle='dashed')
        ax8.plot(T*1000,TOT_t.real,c=(0,0.7,0),linestyle='dashed',linewidth=0.5)
        ax8.plot(T*1000,TOT_t.imag,c=(0.4,0,0.7),linestyle='dashed',linewidth=0.5)
        ax8.set_xlim([-0.43,7])
        ax8.set_ylim([-np.max(abs(self.DPfid)),np.max(abs(self.DPfid))])
        ax8.set_xlabel('time [ms]')
    
    def draw_DP_FID_Fit(self,axd):
        axd.set_title('DP FID Fits')
        T = np.linspace(-0.43e-3,10e-3,1000)
        RBC = self.FIDFitfunction(T,*self.DP_fit_params[0,:])
        MEM = self.FIDFitfunction(T,*self.DP_fit_params[1,:])
        GAS = self.FIDFitfunction(T,*self.DP_fit_params[2,:])
        axd.plot(T*1000,RBC.imag,c=(0.8,0,0),linestyle='dashed',linewidth=0.7)
        axd.plot(T*1000,MEM.imag,c=(0,0.8,0),linestyle='dashed',linewidth=0.7)
        axd.plot(T*1000,GAS.imag,c=(0,0,0.8),linestyle='dashed',linewidth=0.7)
        axd.plot(T*1000,RBC.real,c=(0.8,0,0))
        axd.plot(T*1000,MEM.real,c=(0,0.8,0))
        axd.plot(T*1000,GAS.real,c=(0,0,0.8))
        axd.hlines(y=0,xmin=0,xmax=self.t[-1]*1000,linewidth=0.5,color=(0.3,0.3,0.3),linestyle='dashed')
        axd.vlines(x=0,ymin=-0.6,ymax=0.6,linewidth=0.5,color=(0.3,0.3,0.3),linestyle='dashed')
        axd.set_xlim([-0.43,7])
        axd.set_ylim([-np.max(abs(MEM)),np.max(abs(MEM))])
        axd.set_xlabel('time [ms]')
    
    def draw_DP_spectrum(self,ax9):
        ax9.set_title('DP spectrum')
        w = np.linspace(-0.5,0.5,len(self.DPfid))/self.scanParameters['dwellTime']
        F = np.fft.fftshift(np.fft.fft(self.DPfid))
        ax9.plot(w,abs(F),c=(0,0,0))
        ax9.plot(w,F.real,c=(0,0.7,0))
        ax9.plot(w,F.imag,c=(0.4,0,0.7))
        ax9.set_xlim([-10000,3000])
        ax9.set_xlabel('frequency [Hz]')
    
    def draw_DP_spectra_fit(self,axe):
        axe.set_title('DP Spectra Fits')
        w = np.linspace(-0.5,0.5,len(self.t))/self.scanParameters['dwellTime']
        RBC = self.FIDFitfunction(self.t,*self.DP_fit_params[0,:])
        MEM = self.FIDFitfunction(self.t,*self.DP_fit_params[1,:])
        GAS = self.FIDFitfunction(self.t,*self.DP_fit_params[2,:])
        FRBC = np.fft.fftshift(np.fft.fft(RBC))
        FMEM = np.fft.fftshift(np.fft.fft(MEM))
        FGAS = np.fft.fftshift(np.fft.fft(GAS))
        scalor = np.max(np.concatenate((abs(FGAS),abs(FMEM),abs(FRBC))))
        axe.vlines(x=self.DP_fit_params[0,1],ymin=0,ymax=scalor,linewidth=0.5,color=(1,.3,.3),linestyle='dashed')
        axe.vlines(x=self.DP_fit_params[1,1],ymin=0,ymax=scalor,linewidth=0.5,color=(.3,1,.3),linestyle='dashed')
        axe.vlines(x=self.DP_fit_params[2,1],ymin=0,ymax=scalor,linewidth=0.5,color=(.3,.3,1),linestyle='dashed')
        axe.plot(w,abs(FRBC),c=(0.8,0,0))
        axe.plot(w,abs(FMEM),c=(0,0.8,0))
        axe.plot(w,abs(FGAS),c=(0,0,0.8))
        axe.set_xlim([-10000,3000])
        axe.set_xlabel('frequency [Hz]')
        axe.text(-10000,0.90*scalor,f"  ɷ = ",fontsize=9,fontweight='bold')
        axe.text(-10000,0.80*scalor,f"  ɸ = ",fontsize=9,fontweight='bold')
        axe.text(-10000,0.70*scalor,f"  L = ",fontsize=9,fontweight='bold')
        axe.text(-10000,0.60*scalor,f"  G = ",fontsize=9,fontweight='bold')
        axe.text(-9000,0.90*scalor,f"{np.round(self.DP_fit_params[2,1])} [Hz]",fontsize=9,color=(0,0,0.5),fontweight='bold')
        axe.text(-9000,0.80*scalor,f"{np.round(self.DP_fit_params[2,2])}°",fontsize=9,color=(0,0,0.5),fontweight='bold')
        axe.text(-9000,0.70*scalor,f"{np.round(self.DP_fit_params[2,3])} [Hz]",fontsize=9,color=(0,0,0.5),fontweight='bold')
        axe.text(-9000,0.60*scalor,f"{np.round(self.DP_fit_params[2,4])} [Hz]",fontsize=9,color=(0,0,0.5),fontweight='bold')
        axe.text(-6000,0.90*scalor,f"{np.round(self.DP_fit_params[1,1])} [Hz] ({np.round(   1e6*(self.DP_fit_params[1,1] - self.DP_fit_params[2,1])/self.newGasFrequency,2)})",fontsize=9,color=(0,0.5,0),fontweight='bold')        
        axe.text(-6000,0.80*scalor,f"{np.round(self.DP_fit_params[1,2])}°",fontsize=9,color=(0,0.5,0),fontweight='bold')
        axe.text(-6000,0.70*scalor,f"{np.round(self.DP_fit_params[1,3])} [Hz]",fontsize=9,color=(0,0.5,0),fontweight='bold')
        axe.text(-6000,0.60*scalor,f"{np.round(self.DP_fit_params[1,4])} [Hz]",fontsize=9,color=(0,0.5,0),fontweight='bold')
        axe.text(-500,0.90*scalor,f"{np.round(self.DP_fit_params[0,1])} [Hz] ({np.round(   1e6*(self.DP_fit_params[0,1] - self.DP_fit_params[2,1])/self.newGasFrequency,2)})",fontsize=9,color=(0.5,0,0),fontweight='bold')
        axe.text(-500,0.80*scalor,f"{np.round(self.DP_fit_params[0,2])}°",fontsize=9,color=(0.5,0,0),fontweight='bold')
        axe.text(-500,0.70*scalor,f"{np.round(self.DP_fit_params[0,3])} [Hz]",fontsize=9,color=(0.5,0,0),fontweight='bold')
        axe.text(-500,0.60*scalor,f"{np.round(self.DP_fit_params[0,4])} [Hz]",fontsize=9,color=(0.5,0,0),fontweight='bold')
        axe.text(-6000,0.4*scalor,f"TE90 = {np.round(self.TE90,3)} [ms]",fontsize=9,fontweight='bold')
    
    def draw_GAS_decay(self,axa):
        axa.set_title('Gas Decay')
        gasDecay_fit_function = lambda x, a, b, c: a * np.cos(b) ** (x - 1) + c
        xdata = np.arange(1, len(self.gasDecay) + 1)
        axa.plot(xdata, gasDecay_fit_function(xdata, *self.flipAngleFitParams), 'r', label='Fit')
        axa.plot(xdata, self.gasDecay, 'bo', markerfacecolor='b', label='Acquired')
        axa.text(np.max(xdata)*0.2,np.max(self.gasDecay)*1.0,f"New Gas Frequency: {np.round(self.newGasFrequency)} [Hz]",fontsize=10.5)
        axa.text(np.max(xdata)*0.3,np.max(self.gasDecay)*0.95,f"Calculated FA: {np.round(self.flip_angle,1)}°±{np.round(self.flip_err,1)}°",fontsize=10.5)
        axa.text(np.max(xdata)*0.3,np.max(self.gasDecay)*0.90,f"New Ref Voltage: {np.round(self.newVoltage)} [V]",fontsize=10.5)
        axa.text(np.max(xdata)*0.3,np.max(self.gasDecay)*0.85,f"TE90: {np.round(self.TE90,3)} [ms]",fontsize=10.5)
        axa.set_title(f"Flip Cal: V_ref = {self.scanParameters['referenceVoltage']}, FA = 20°")
    
    def draw_wiggles(self,axb):
        try:
            axb.set_title('Wiggles')
            axb.hlines(y=np.arange(0,1,0.1),xmin=np.repeat(0,10),xmax=np.repeat(10,10),color = (0.8,0.8,0.8),linestyle='dashed',linewidth=0.5)
            axb.plot(np.linspace(100*int(self.scanParameters['TR'])*1e-6,int(self.scanParameters['TR'])*len(self.RBC2MEMsig_wiggles)*1e-6,len(self.RBC2MEMsig_wiggles[100:])), self.RBC2MEMsig_wiggles[100:],color='#0000ff')
            axb.plot(np.linspace(100*int(self.scanParameters['TR'])*1e-6,int(self.scanParameters['TR'])*len(self.RBC2MEMmag_wiggles)*1e-6,len(self.RBC2MEMmag_wiggles[100:])), self.RBC2MEMmag_wiggles[100:],color='#ff0000')
            # axb.plot(np.linspace(100*int(self.scanParameters['TR'])*1e-6,int(self.scanParameters['TR'])*len(self.RBC2MEMdix_wiggles)*1e-6,len(self.RBC2MEMdix_wiggles[100:])), self.RBC2MEMdix_wiggles[100:],color='#00ff00')
            axb.set_ylim([0,1])
            axb.set_xlim([100*int(self.scanParameters['TR'])*1e-6,len(self.RBC2MEMmag_wiggles)*int(self.scanParameters['TR'])*1e-6])
            axb.set_title(f"RBC/MEM vs Time")
            axb.text(2,0.95,f"RBC/MEM signal = {np.round(np.mean(self.RBC2MEMsig_wiggles[100:]),3)}",fontsize=11,color='#0000ff')
            axb.text(2,0.90,f"RBC/MEM magnitude = {np.round(np.mean(self.RBC2MEMmag_wiggles[100:]),3)}",fontsize=11,color='#ff0000')
            # axb.text(2,0.85,f"RBC/MEM dixon = {np.round(np.mean(self.RBC2MEMdix_wiggles[100:]),3)}",fontsize=11,color='#00ff00')
            # axb.text(2,0.80,f"RBC/MEM amp = {np.round(self.RBC2MEMmag_amp,3)} = {np.round(200*self.RBC2MEMmag_amp/self.RBC2MEMmag,2)} %",fontsize=12)
        except:
            print(f"No Wiggles to print")
            axb.text(0.5,0.5,f"Wiggles not processed",fontsize=12)
    
    def printout(self,save_path = None):
        '''Summarizes Flipcal in a png and prints important stuff to console'''
        ## --- CREATE FIGURE--- ##
        if save_path is None:
            save_path = f"C:/PIRL/data/FlipCal_{self.patientInfo['PatientName']}.png"
        fig = plt.figure(constrained_layout=True)
        fig.set_size_inches(18,9)
        gs = fig.add_gridspec(4, 8, left=0.05, right=0.95, hspace=0.5, wspace=0.5)
        ax1 = fig.add_subplot(gs[0,0:2])
        ax1.plot([0,1],[0,1],c=(1,1,1))
        ## --- FILE INFO --- ## Requires self.patientInfo from __init__
        ax1.text(0,1.00,f"Patient Name: ",fontsize=11);ax1.text(0.4,1.00,f"{self.patientInfo['PatientName']}",fontsize=12)
        ax1.text(0,0.85,f"Scan Date:    ",fontsize=12);ax1.text(0.4,0.85,f"{self.scanParameters['scanDate']}",fontsize=12)
        ax1.text(0,0.70,f"Gas Frequency:",fontsize=12);ax1.text(0.4,0.7,f"{self.scanParameters['GasFrequency']}",fontsize=12)
        ax1.set_axis_off()
        ## --- NOISE --- ## Requires self.FID from __init__
        ax2 = fig.add_subplot(gs[0,4])
        noiseData = np.concatenate((self.FID[:,0].real,self.FID[:,0].imag))
        synthData = np.random.normal(loc = np.mean(noiseData),scale = np.std(noiseData), size = len(noiseData))
        ax2.plot(np.linspace(np.min(noiseData),np.max(noiseData),10),np.linspace(np.min(noiseData),np.max(noiseData),10),c=(1,0,0))
        ax2.scatter(np.sort(noiseData),np.sort(synthData),s=0.5,color=(0,0,0))
        ax2.set_title("Noise QQ Plot")
        ax3 = fig.add_subplot(gs[0, 5])
        ax3.set_title('Noise Hist')
        ax3.hist(noiseData,30,range=(-1e-5,1e-5),color=(.4,.7,1))
        ax3.vlines(x = [-np.std(noiseData),np.std(noiseData)],ymin=[0,0],ymax=[100,100], color = (0.3,0.5,0.8),linestyle='dashed')
        ## --- GAS FID (phasor) --- ## REquires self.gas_fit_params from fit_GAS_FID()
        ax4 = fig.add_subplot(gs[1, 0])
        self.draw_GAS_phasor(ax4)
        ## --- GAS FID (FID) --- ## REquires self.GASfid from __init__ and self.gas_fit_params from fit_GAS_FID()
        ax5 = fig.add_subplot(gs[1, 1:3])
        self.draw_GAS_FID(ax5)
        ## --- GAS FID (SPECTRUM) --- ## REquires self.newGasFrequency from fit_GAS_FID()
        ax6 = fig.add_subplot(gs[1, 3:6])
        self.draw_GAS_spectrum(ax6)
        ## ------- DP FID (phasor) -------# (requires self.DP_fit_params)
        TE = int(self.scanParameters['TE'])*1e-6
        RBC_n0 = self.FIDFitfunction(0, *self.DP_fit_params[0,:])
        MEM_n0 = self.FIDFitfunction(0, *self.DP_fit_params[1,:])
        GAS_n0 = self.FIDFitfunction(0, *self.DP_fit_params[2,:])
        TOT_n0 = RBC_n0 + MEM_n0 + GAS_n0
        RBC_nTE = self.FIDFitfunction(-TE, *self.DP_fit_params[0,:])
        MEM_nTE = self.FIDFitfunction(-TE, *self.DP_fit_params[1,:])
        GAS_nTE = self.FIDFitfunction(-TE, *self.DP_fit_params[2,:])
        TOT_nTE = RBC_nTE + MEM_nTE + GAS_nTE
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.set_title('DP Phasor (t=0)')
        ax7.plot([0,1],[0,1],c=(1,1,1))
        ax7.vlines(x = [0],ymin=[-1],ymax=[1], color = (0.3,0.5,0.8),linestyle='dashed')
        ax7.hlines(y = [0],xmin=[-1],xmax=[1], color = (0.3,0.5,0.8),linestyle='dashed')
        ax7.arrow(0,0,TOT_nTE.real/abs(TOT_nTE),TOT_nTE.imag/abs(TOT_nTE),color=(0,0,0),head_width=0.05, head_length=0.1,linewidth=1,linestyle='dashed')
        ax7.arrow(0,0,TOT_n0.real/abs(TOT_n0),TOT_n0.imag/abs(TOT_n0),color=(0,0,0),head_width=0.05, head_length=0.1,linewidth=2)
        ax7.set_xlim([-1,1])
        ax7.set_axis_off()
        ## ------- DP FID (FID) -------#
        ax8 = fig.add_subplot(gs[2, 1:3])
        self.draw_DP_FID(ax8)
        ## ------- DP FID (spectrum) -------#
        ax9 = fig.add_subplot(gs[2, 3:6])
        self.draw_DP_spectrum(ax9)
        ## ------- DP FITS (phasor)-------#
        axc = fig.add_subplot(gs[3, 0])
        self.draw_DP_phasor(axc)
        ## ------- DP FITS FID-------#
        axd = fig.add_subplot(gs[3, 1:3])
        self.draw_DP_FID_Fit(axd)
        ## ------- DP FITS Spectra -------#
        axe = fig.add_subplot(gs[3, 3:6])
        self.draw_DP_spectra_fit(axe)
        ### ----- Gas Decay and Flip Angle Fit ------ ###
        axa = fig.add_subplot(gs[0:2, 6:8])
        self.draw_GAS_decay(axa)
        ### ----- Wiggles ------ ###
        axb = fig.add_subplot(gs[2:4, 6:8])
        self.draw_wiggles(axb)
        #plt.show()
        print(f"\033[36mConsole printout for \033[32m{self.patientInfo['PatientName']}\033[36m imaged at \033[32m{self.scanParameters['scanDate']} \033[37m")
        print(f"\033[36mGas Frequency should be set to \033[32m{np.round(self.newGasFrequency,0)}\033[37m")
        print(f"\033[36mSet the Ref Voltage to \033[32m{np.round(self.newVoltage,0)}\033[37m")
        print(f"\033[36mTE90 is \033[32m{np.round(self.TE90,3)}\033[37m")
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.3, hspace=0.3)
        plt.savefig(save_path)
        print(f"\033[36mPrintout saved to \033[33m{save_path}\033[37m")
    
    def dicomPrintout(self,dummy_dicom_path = None,save_path = 'c:/pirl/data/'):
        '''Creates a 3D DICOM file (enhanced) where each image is a matplotlib pyplot.
        For each plot we want to dicomize, we create a plot then convert to numpy array.
        Then each array is stacked and a DICOM file is created (need to try this on PACS)'''
        if dummy_dicom_path is None:
            print("Argument 'dummy_dicom_path' is not specified for dicomPrintout(), aborting DICOM printout")
            return
        # -- 1 - Gas Decay and Flip Angle
        fig_size = (7,4)
        fig, axa = plt.subplots(figsize = fig_size)
        self.draw_GAS_decay(axa)
        fig.canvas.draw()
        GAS_Decay_Fit = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        GAS_Decay_Fit = GAS_Decay_Fit.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        # -- 2 -GAS FID -- #
        fig, ax5 = plt.subplots(figsize = fig_size)
        self.draw_GAS_FID(ax5)
        fig.canvas.draw()
        GAS_FID_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        GAS_FID_fig = GAS_FID_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        # -- 3 -GAS SPECTRUM -- #
        fig, ax5 = plt.subplots(figsize = fig_size)
        self.draw_GAS_spectrum(ax5)
        fig.canvas.draw()
        GAS_SPECTRUM_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        GAS_SPECTRUM_fig = GAS_SPECTRUM_fig.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        # -- 4 -DP FID Fit -- #
        fig, axe = plt.subplots(figsize = fig_size)
        self.draw_DP_FID(axe)
        fig.canvas.draw()
        DP_FID = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        DP_FID = DP_FID.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        # -- 5 -DP FID Fit -- #
        fig, axe = plt.subplots(figsize = fig_size)
        self.draw_DP_FID_Fit(axe)
        fig.canvas.draw()
        DP_FID_FIT = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        DP_FID_FIT = DP_FID_FIT.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        # -- 6 -DP Spectra Fit-- #
        fig, axe = plt.subplots(figsize = fig_size)
        self.draw_DP_spectra_fit(axe)
        fig.canvas.draw()
        DP_SPECTRA_FIT = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        DP_SPECTRA_FIT = DP_SPECTRA_FIT.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        # - 7 -Wiggles - #
        fig, axb = plt.subplots(figsize = fig_size)
        try:
            self.draw_wiggles(axb) 
        except:
            axb.text(0.5,0.5,f"Wiggles not processed",fontsize=12)
        fig.canvas.draw()
        WIGGLES = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        WIGGLES = WIGGLES.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        # -- Build Array -- ##
        try:
            image_data = np.stack((GAS_Decay_Fit,GAS_FID_fig,GAS_SPECTRUM_fig,DP_FID,DP_FID_FIT,DP_SPECTRA_FIT,WIGGLES),axis=0)
        except:
            print(f"Array Shapes don't match for DICOM export for some reason...")
        # Load the template DICOM file
        template_dicom = pydicom.dcmread(dummy_dicom_path)
        # Extract patient and scan metadata
        def copy_metadata(src_dcm):
            new_dcm = Dataset()
            for elem in src_dcm.iterall():
                if elem.tag not in [0x7FE00010]:  # Exclude Pixel Data
                    new_dcm.add(elem)
            return new_dcm
        # Ensure output folder exists
        os.makedirs(save_path, exist_ok=True)
        # Generate a single SeriesInstanceUID to ensure all images stay in the same series
        shared_series_uid = pydicom.uid.generate_uid()
        for image_set_index in range(image_data.shape[0]):
            pixel_array = np.array(image_data[image_set_index,:,:,:], dtype=np.uint8)
            dicom_file = copy_metadata(template_dicom)
            # Assign unique values
            dicom_file.SOPInstanceUID = pydicom.uid.generate_uid()  # Unique for each image
            dicom_file.InstanceNumber = image_set_index+1 # Order the pages correctly
            # Keep these the same for all pages
            dicom_file.StudyInstanceUID = template_dicom.StudyInstanceUID  # Same study
            dicom_file.SeriesInstanceUID = shared_series_uid  # Same series for all pages
            dicom_file.SeriesNumber = 999  # Consistent series number
            dicom_file.SeriesDescription = f"FlipCal_{self.patientInfo['PatientName']}"  # Custom label
            dicom_file.ImageType = ["DERIVED", "SECONDARY"]
            dicom_file.ContentDate = datetime.datetime.now().strftime("%Y%m%d")
            dicom_file.ContentTime = datetime.datetime.now().strftime("%H%M%S")
            dicom_file.Manufacturer = "MU PIRL version 250312"
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
            dicom_file.is_little_endian = template_dicom.is_little_endian
            dicom_file.is_implicit_VR = template_dicom.is_implicit_VR
            # Save the new DICOM file
            output_path = os.path.join(save_path, f"FlipCal_{image_set_index:03d}.dcm")
            dicom_file.save_as(output_path)
            print(f"Saved: {output_path}")
        # - Create and save a dicom - #
        # file_meta = dicom.dataset.FileMetaDataset()
        # file_meta.MediaStorageSOPClassUID = dicom.uid.generate_uid()
        # file_meta.MediaStorageSOPInstanceUID = dicom.uid.generate_uid()
        # file_meta.ImplementationClassUID = dicom.uid.generate_uid()
        # ds = dicom.dataset.FileDataset("output.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
        # ds.PatientName = self.patientInfo['PatientName']
        # ds.PatientID = self.patientInfo['PatientID']
        # ds.StudyInstanceUID = dicom.uid.generate_uid()
        # ds.SeriesInstanceUID = dicom.uid.generate_uid()
        # ds.SOPInstanceUID = dicom.uid.generate_uid()
        # ds.Modality = "MR"
        # ds.StudyDate = datetime.datetime.now().strftime("%Y%m%d")
        # ds.StudyTime = datetime.datetime.now().strftime("%H%M%S")
        # ds.Manufacturer = self.scanParameters['systemVendor']
        # # Image data specifics for RGB
        # ds.NumberOfFrames, ds.Rows, ds.Columns, _ = image_data.shape
        # ds.SamplesPerPixel = 3  # RGB
        # ds.PhotometricInterpretation = "RGB"
        # ds.PlanarConfiguration = 0  # 0: RGBRGB... (interleaved), 1: RRR...GGG...BBB...
        # ds.BitsAllocated = 8  # 8 bits per channel
        # ds.BitsStored = 8
        # ds.HighBit = 7
        # ds.PixelRepresentation = 0  # Unsigned integer
        # ds.PixelData = image_data.tobytes()
        # # Save to file
        # ds.save_as(save_path)
    
    def pickleMe(self, pickle_path='C:/PIRL/data/FlipCalPickle.pkl'):
        '''Uses dictionary comprehension to create a dictionary of all class attributes, then saves as pickle'''
        pickle_dict = {}
        for attr in vars(self):
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
            setattr(self, attr, value)
    
    def exportISMRMRD(self,path='c:/pirl/data/ISMRMRD.h5'):
            '''Consortium-Required Parameters: https://github.com/Xe-MRI-CTC/siemens-to-mrd-converter'''
            if os.path.exists(path):
                os.remove(path)
            ismrmrd_header = ismrmrd.xsd.ismrmrdHeader()
            ismrmrd_header.studyInformation = ismrmrd_header.studyInformation or ismrmrd.xsd.studyInformationType()
            ismrmrd_header.subjectInformation = ismrmrd_header.subjectInformation or ismrmrd.xsd.subjectInformationType()
            ismrmrd_header.acquisitionSystemInformation = ismrmrd_header.acquisitionSystemInformation or ismrmrd.xsd.acquisitionSystemInformationType()
            ismrmrd_header.sequenceParameters = ismrmrd_header.sequenceParameters or ismrmrd.xsd.sequenceParametersType()
            ismrmrd_header.studyInformation.studyDate = self.scanParameters['scanDate'] #scan_date
            ismrmrd_header.subjectInformation.patientID = self.patientInfo['PatientName']#subject_id
            ismrmrd_header.acquisitionSystemInformation.systemVendor = self.scanParameters['systemVendor']#system_vendor
            ismrmrd_header.acquisitionSystemInformation.institutionName = self.scanParameters['institutionName']
            ismrmrd_header.acquisitionSystemInformation.systemFieldStrength_T = self.scanParameters['B0fieldStrength']
            ismrmrd_header.sequenceParameters.TE = self.scanParameters['TE']
            ismrmrd_header.sequenceParameters.TR = self.scanParameters['TR']
            ismrmrd_header.sequenceParameters.flipAngle_deg = self.scanParameters['FlipAngle']
            # -- user defined attributes here (non-native ISMRMRD header stuffs that xenon people like)
            # -- for Calibration this includes Gas frequency and DP offset frequency
            ismrmrd_header.userParameters = ismrmrd_header.userParameters or ismrmrd.xsd.userParametersType()
            # if not hasattr(ismrmrd_header.userParameters, 'userParameterLong'):
            #     ismrmrd_header.userParameters.userParameterLong = []
            ismrmrd_header.userParameters.userParameterLong.append(ismrmrd.xsd.userParameterLongType("xe_center_frequency",self.scanParameters['GasFrequency']))
            ismrmrd_header.userParameters.userParameterLong.append(ismrmrd.xsd.userParameterLongType("xe_dissolved_offset_frequency",self.scanParameters['DisFrequencyOffset']))
            if not hasattr(ismrmrd_header.encoding, "trajectoryDescription"):
                ismrmrd_header.encoding = ismrmrd.xsd.encodingType()
            if type(ismrmrd_header.encoding.reconSpace) == type(None):
                ismrmrd_header.encoding.reconSpace = ismrmrd.xsd.encodingSpaceType()
            fov_obj = ismrmrd.xsd.fieldOfViewMm()
            fov_obj.x = self.scanParameters['FOV']
            fov_obj.y = self.scanParameters['FOV']
            fov_obj.z = self.scanParameters['FOV']
            ismrmrd_header.encoding.reconSpace.fieldOfView_mm = fov_obj
            # -- write the actual FID data one FID at a time
            ismrmrd_data_set = ismrmrd.Dataset(path, "/dataset", create_if_needed=True)
            for acquisition_num in range(self.FID.shape[1]):
                acquisition = ismrmrd.Acquisition()
                acquisition_header = ismrmrd.AcquisitionHeader()
                acquisition_header.number_of_samples = self.FID.shape[0]
                acquisition_header.active_channels = 1
                acquisition_header.trajectory_dimensions = 3
                acquisition_header.sample_time_us = self.scanParameters['dwellTime']*1e6
                acquisition_header.idx.contrast = int(2*(acquisition_num<500) + 1*(acquisition_num>=500))
                acquisition_header.measurement_uid = int(0) #Bonus Spectra (not in calibration)
                acquisition.resize(512)
                acquisition.version = 1
                acquisition.available_channels = 1
                acquisition.center_sample = 0
                acquisition.read_dir[0] = 1.0
                acquisition.phase_dir[1] = 1.0
                acquisition.slice_dir[2] = 1.0
                acquisition.setHead(acquisition_header)
                acquisition.data[:] = self.FID[:,acquisition_num]
                ismrmrd_data_set.append_acquisition(acquisition)
            ismrmrd_data_set.write_xml_header(ismrmrd.xsd.ToXML(ismrmrd_header))
            ismrmrd_data_set.close()
            print(f'ISMRMRD h5 file written to {path}')
    
    def extract_attributes(self, attr_dict, parent_key='', sep='_'):
        """Helper method which creates a single dictionary from an attribute dictionary"""
        items = []
        for k, v in attr_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                # If the value is a dictionary, recurse
                items.extend(self.extract_attributes(v, new_key, sep=sep).items())
            else:
                # Otherwise, add the attribute to the items list
                items.append((new_key, v))
        return dict(items)
    
    def twix_header_to_json(self,output_path='c:/tmp/TWIXheader.json'):
        twix_header = self.twix.hdr
        with open(output_path, 'w') as fp:
            json.dump(extract_attributes(twix_header), fp,indent=2)
        return twix_header
    
    def __repr__(self):
        string = (f'\033[35mFlipCal\033[37m class object version \033[94m{self.version}\033[37m\n')
        for attr, value in vars(self).items():
            if value == '':
                string += (f'\033[31m {attr}: \033[37m\n')
            elif attr == 'calibration_dict':
                string += (f'\033[32m {attr}: \033[33mexists\033[37m\n')
            elif attr == 'gas_fit_params':
                string += "\033[32mgas_fit_params\033[36m  Frequency -- Area -- Phase -- FWHML -- FWHMG\033[37m \n"
                string += f"                \033[37m {np.round(self.gas_fit_params[1])} -- {np.round(self.gas_fit_params[0],3)} -- {np.round(self.gas_fit_params[2])} -- {np.round(self.gas_fit_params[3])} -- {np.round(self.gas_fit_params[4])}\n"
            elif attr == 'DP_fit_params':
                string += "\033[32mDP_fit_params\033[36m     Frequency -- Area -- Phase -- FWHML -- FWHMG\033[37m \n"
                string += f"            \033[36mRBC:\033[37m {np.round(self.DP_fit_params[0,1])} -- {np.round(self.DP_fit_params[0,0]/self.DP_fit_params[1,0],3)} -- {np.round(self.DP_fit_params[0,2])} -- {np.round(self.DP_fit_params[0,3])} -- {np.round(self.DP_fit_params[0,4])}\n"
                string += f"            \033[36mMEM:\033[37m {np.round(self.DP_fit_params[1,1])} -- {np.round(self.DP_fit_params[1,0]/self.DP_fit_params[1,0],3)} -- {np.round(self.DP_fit_params[1,2])} -- {np.round(self.DP_fit_params[1,3])} -- {np.round(self.DP_fit_params[1,4])}\n"
                string += f"            \033[36mGAS:\033[37m {np.round(self.DP_fit_params[2,1])} -- {np.round(self.DP_fit_params[2,0]/self.DP_fit_params[1,0],3)} -- {np.round(self.DP_fit_params[2,2])} -- {np.round(self.DP_fit_params[2,3])} -- {np.round(self.DP_fit_params[2,4])}\n"
            elif type(value) is np.ndarray:
                string += (f'\033[32m {attr}: \033[36m{value.shape} \033[37m\n')
            elif attr == 'scanParameters':
                string += '\033[35mscanParameters\033[37m \n'
                for attr2, value2 in value.items():
                    if value2 == '':
                        string += (f'   \033[31m {attr2}: \033[37m\n')
                    else:
                        string += (f'   \033[32m {attr2}: \033[36m{value2} \033[37m\n')
            elif attr == 'patientInfo':
                string += '\033[35mpatientInfo\033[37m \n'
                for attr2, value2 in value.items():
                    if value2 == '':
                        string += (f'   \033[31m {attr2}: \033[37m\n')
                    else:
                        string += (f'   \033[32m {attr2}: \033[36m{value2} \033[37m\n')
            elif type(value) is dict:
                string += (f'\033[35m{attr} also found\033[37m \n')
            else:
                string += (f'\033[32m {attr}: \033[36m{value} \033[37m\n')
        return string



### --------------------------------------------------------------------------------------------####
### -----------------------------------------Main GUI Script -----------------------------------####
### --------------------------------------------------------------------------------------------####
if __name__ == "__main__":
    version = '250128_calibrationGUI'
    image_box_size = 50
    ARCHIVE_path = '//umh.edu/data/Radiology/Xenon_Studies/Studies/Archive/'
    import PySimpleGUI as sg
    from datetime import date # -- So we can export the analysis date
    from PIL import Image, ImageTk, ImageDraw, ImageFont # ---------- for arrayToImage conversion
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
    
    patient_data_column = [[sg.Button('',                      pad=(0,0)),sg.Text('Calibration version:      ',key='FAversion',   pad=(0,0))],
                           [sg.Button('',key='editPatientName',pad=(0,0)),sg.Text('Subject:         ',key='subject',     pad=(0,0))],
                           [sg.Button('',key='editStudyDate',  pad=(0,0)),sg.Text('Study Date:      ',key='date',        pad=(0,0))],
                           [sg.Button('',                      pad=(0,0)),sg.Text('Gas Frequency:   ',key='gasFrequency',pad=(0,0))],
                           [sg.Button('',                      pad=(0,0)),sg.Text('Dwell Time:      ',key='dwellTime',   pad=(0,0))],
                           [sg.Button('',                      pad=(0,0)),sg.Text('Protocol:        ',key='twixprotocol',pad=(0,0))],
                           [sg.Button('',                      pad=(0,0)),sg.Text('New Frequency:   ',key='newFrequency',pad=(0,0))],
                           [sg.Button('',                      pad=(0,0)),sg.Text('New Voltage:     ',key='newVoltage',  pad=(0,0))],
                           [sg.Button('',                      pad=(0,0)),sg.Text('TE90:            ',key='TE90',        pad=(0,0))],
                           [sg.Button('',                      pad=(0,0)),sg.Text('RBC2MEMsig:      ',key='RBC2MEMsig',  pad=(0,0))],
                           [sg.Button('',                      pad=(0,0)),sg.Text('RBC2MEMmag:      ',key='RBC2MEMmag',  pad=(0,0))],
                           [sg.Button('',                      pad=(0,0)),sg.Text('RBC2MEMdix:      ',key='RBC2MEMdix',  pad=(0,0))],
                           [sg.Button('',key='editDE',         pad=(0,0)),sg.Text('DE:              ',key='DE',          pad=(0,0))]]
    
    windowLayout = [[sg.Text('Calibration File'),sg.InputText(key='filepath',default_text="C:/PIRL/data/Afia/meas_MID00083_FID101502_5_Xe_fid_calibration_dyn.dat",size=(100,1)),sg.Button('Load',key='LoadFile')],
                    [sg.Radio('Twix','filetype',key='twixfile',default=True),sg.Radio('Pickle','filetype',key='picklefile'),sg.Radio('ISMRMRD','filetype',key='ismrmrdfile'),sg.Radio('Matlab','filetype',key='matlabfile')],
                   [sg.Column(patient_data_column),sg.Canvas(key='-GASDECAY-'),sg.Canvas(key='-DPPLOT-')],
                   [sg.Canvas(key='-WIGGLES-')],
                   [sg.Button('Process FlipCal',key='process'),sg.Button('Process Wiggles',key='wiggles')],
                   [sg.Text('Dummy Dicom Path'),sg.InputText(key='dummy_dicom_path',size=(100,1))],
                   [sg.Text('Save Directory'),sg.InputText(key='SAVEpath',default_text='C:/PIRL/data/FA/',size=(100,1)),sg.Button('Save',key='savedata')]]
    
    window = sg.Window(f'PIRL FlipCal Analysis -- {version}', windowLayout, return_keyboard_events=True, margins=(0, 0), finalize=True, size= (1000,550),resizable=True)
    def draw_figure(canvas, figure):
        figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
        figure_canvas_agg.draw()
        figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
        return figure_canvas_agg
    def updateData():
        if 'FA' in globals():
            try:
                window['FAversion'].update(f"Calibration version: {FA.version}")
                window['subject'].update(f"Subject: {FA.patientInfo['PatientName']}")
                window['date'].update(f"Date: {FA.scanParameters['scanDate']}")
                window['gasFrequency'].update(f"Gas Frequency = {FA.scanParameters['GasFrequency']} Hz")
                window['dwellTime'].update(f"Dwell Time = {FA.scanParameters['dwellTime']} us")
                window['twixprotocol'].update(f"Protocol = {FA.scanParameters['ProtocolName']}")
                window['newFrequency'].update(f"New Frequency = {np.round(FA.newGasFrequency,1)} Hz",font=('bold'))
                window['newVoltage'].update(f"New Voltage = {np.round(FA.newVoltage,1)} V",font=('bold'))
                window['TE90'].update(f"TE90 = {np.round(FA.TE90,3)} us",font=('bold'))
                window['RBC2MEMsig'].update(f"RBC2MEMsig = {np.round(FA.RBC2MEMsig,3)}",font=('bold'))
                window['RBC2MEMmag'].update(f"RBC2MEMmag = {np.round(FA.RBC2MEMmag,3)}",font=('bold'))
                window['RBC2MEMdix'].update(f"RBC2MEMdix = {np.round(FA.RBC2MEMdix,3)}",font=('bold'))
                window['DE'].update(f"DE = {FA.patientInfo['DE']} mL",font=('bold'))
            except Exception as e:
                print(e)
    def updateDecay():
            gasDecay_fit_function = lambda x, a, b, c: a * np.cos(b) ** (x - 1) + c
            xdata = np.arange(1, len(FA.gasDecay) + 1)
            plt.figure(figsize=(3,3))
            plt.plot(xdata, gasDecay_fit_function(xdata, *FA.flipAngleFitParams), 'r', label='Fit',linewidth=0.5)
            plt.plot(xdata, FA.gasDecay, 'bo', markerfacecolor='b', label='Acquired',markersize=2)
            plt.text(np.max(xdata)*0.1,np.max(FA.gasDecay)*1.0,f"New Gas Frequency: {np.round(FA.newGasFrequency)} [Hz]",fontsize=8.5)
            plt.text(np.max(xdata)*0.3,np.max(FA.gasDecay)*0.95,f"Calculated FA: {np.round(FA.flip_angle,1)}°±{np.round(FA.flip_err,1)}°",fontsize=8.5)
            try:
                plt.text(np.max(xdata)*0.3,np.max(FA.gasDecay)*0.90,f"New Ref Voltage: {np.round(FA.newVoltage)} [V]",fontsize=8.5)
            except:
                plt.text(np.max(xdata)*0.3,np.max(FA.gasDecay)*0.90,f"No ref voltage to scale to",fontsize=8.5)
            plt.text(np.max(xdata)*0.3,np.max(FA.gasDecay)*0.85,f"TE90: {np.round(FA.TE90,3)} [ms]",fontsize=8.5)
            plt.text(np.max(xdata)*0.3,np.max(FA.gasDecay)*0.80,f"RBC2MEMsig: {np.round(FA.RBC2MEMsig,3)}",fontsize=8.5)
            plt.text(np.max(xdata)*0.3,np.max(FA.gasDecay)*0.75,f"RBC2MEMmag: {np.round(FA.RBC2MEMmag,3)}",fontsize=8.5)
            plt.text(np.max(xdata)*0.3,np.max(FA.gasDecay)*0.70,f"RBC2MEMdix: {np.round(FA.RBC2MEMdix,3)}",fontsize=8.5)
            draw_figure(window['-GASDECAY-'].TKCanvas,plt.gcf())
    def updateDP():
        plt.figure(figsize=(6,2.5))
        w = np.linspace(-0.5,0.5,len(FA.t))/FA.scanParameters['dwellTime']
        RBC = FA.FIDFitfunction(FA.t,*FA.DP_fit_params[0,:])
        MEM = FA.FIDFitfunction(FA.t,*FA.DP_fit_params[1,:])
        GAS = FA.FIDFitfunction(FA.t,*FA.DP_fit_params[2,:])
        FRBC = np.fft.fftshift(np.fft.fft(RBC))
        FMEM = np.fft.fftshift(np.fft.fft(MEM))
        FGAS = np.fft.fftshift(np.fft.fft(GAS))
        scalor = np.max(np.concatenate((abs(FGAS),abs(FMEM),abs(FRBC))))
        plt.vlines(x=FA.DP_fit_params[0,1],ymin=0,ymax=scalor,linewidth=0.5,color=(1,.3,.3),linestyle='dashed')
        plt.vlines(x=FA.DP_fit_params[1,1],ymin=0,ymax=scalor,linewidth=0.5,color=(.3,1,.3),linestyle='dashed')
        plt.vlines(x=FA.DP_fit_params[2,1],ymin=0,ymax=scalor,linewidth=0.5,color=(.3,.3,1),linestyle='dashed')
        plt.plot(w,abs(FRBC),c=(0.8,0,0))
        plt.plot(w,abs(FMEM),c=(0,0.8,0))
        plt.plot(w,abs(FGAS),c=(0,0,0.8))
        plt.xlim((-10000,3000))
        plt.text(-10000,0.90*scalor,f"  ɷ = ",fontsize=8)
        plt.text(-10000,0.75*scalor,f"  ɸ = ",fontsize=8)
        plt.text(-10000,0.60*scalor,f"  L = ",fontsize=8) 
        plt.text(-10000,0.45*scalor,f"  G = ",fontsize=8)
        plt.text(-9000,0.90*scalor,f"{np.round(FA.DP_fit_params[2,1])} [Hz]",fontsize=8,color=(0,0,0.5))
        plt.text(-9000,0.75*scalor,f"{np.round(FA.DP_fit_params[2,2])}°",fontsize=8,color=(0,0,0.5))
        plt.text(-9000,0.60*scalor,f"{np.round(FA.DP_fit_params[2,3])} [Hz]",fontsize=8,color=(0,0,0.5))
        plt.text(-9000,0.45*scalor,f"{np.round(FA.DP_fit_params[2,4])} [Hz]",fontsize=8,color=(0,0,0.5))
        plt.text(-6500,0.90*scalor,f"{np.round(FA.DP_fit_params[1,1])} [Hz] ({np.round(   1e6*(FA.DP_fit_params[1,1] - FA.DP_fit_params[2,1])/FA.scanParameters['GasFrequency'],2)})",fontsize=8,color=(0,0.5,0))
        plt.text(-6500,0.75*scalor,f"{np.round(FA.DP_fit_params[1,2])}°",fontsize=8,color=(0,0.5,0))
        plt.text(-6500,0.60*scalor,f"{np.round(FA.DP_fit_params[1,3])} [Hz]",fontsize=8,color=(0,0.5,0))
        plt.text(-6500,0.45*scalor,f"{np.round(FA.DP_fit_params[1,4])} [Hz]",fontsize=8,color=(0,0.5,0))
        plt.text(-3000,0.90*scalor,f"{np.round(FA.DP_fit_params[0,1])} [Hz] ({np.round(   1e6*(FA.DP_fit_params[0,1] - FA.DP_fit_params[2,1])/FA.scanParameters['GasFrequency'],2)})",fontsize=8,color=(0.5,0,0))
        plt.text(-3000,0.75*scalor,f"{np.round(FA.DP_fit_params[0,2])}°",fontsize=8,color=(0.5,0,0))
        plt.text(-3000,0.60*scalor,f"{np.round(FA.DP_fit_params[0,3])} [Hz]",fontsize=8,color=(0.5,0,0))
        plt.text(-3000,0.45*scalor,f"{np.round(FA.DP_fit_params[0,4])} [Hz]",fontsize=8,color=(0.5,0,0))
        draw_figure(window['-DPPLOT-'].TKCanvas,plt.gcf())    
    def updateWiggles():
        plt.figure(figsize=(6,2.5))
        plt.hlines(y=np.arange(0,1,0.1),xmin=np.repeat(0,10),xmax=np.repeat(10,10),color = (0.8,0.8,0.8),linestyle='dashed',linewidth=0.5)
        plt.plot(np.linspace(100*int(FA.scanParameters['TR'])*1e-6,int(FA.scanParameters['TR'])*len(FA.RBC2MEMsig_wiggles)*1e-6,len(FA.RBC2MEMsig_wiggles[100:])), FA.RBC2MEMsig_wiggles[100:])
        plt.plot(np.linspace(100*int(FA.scanParameters['TR'])*1e-6,int(FA.scanParameters['TR'])*len(FA.RBC2MEMmag_wiggles)*1e-6,len(FA.RBC2MEMmag_wiggles[100:])), FA.RBC2MEMmag_wiggles[100:])
        plt.plot(np.linspace(100*int(FA.scanParameters['TR'])*1e-6,int(FA.scanParameters['TR'])*len(FA.RBC2MEMdix_wiggles)*1e-6,len(FA.RBC2MEMdix_wiggles[100:])), FA.RBC2MEMdix_wiggles[100:])
        plt.ylim([0,1])
        plt.xlim([100*int(FA.scanParameters['TR'])*1e-6,len(FA.RBC2MEMsig_wiggles)*int(FA.scanParameters['TR'])*1e-6])
        plt.title(f"RBC/MEM vs Time")
        plt.text(2,0.90,f"RBC/MEMsig mean = {np.round(np.mean(FA.RBC2MEMsig_wiggles[100:]),3)}",fontsize=12)
        plt.text(2,0.82,f"RBC/MEMmag mean = {np.round(np.mean(FA.RBC2MEMmag_wiggles[100:]),3)}",fontsize=12)
        plt.text(2,0.74,f"RBC/MEMdix mean = {np.round(np.mean(FA.RBC2MEMdix_wiggles[100:]),3)}",fontsize=12)
        #plt.text(2,0.66,f"RBC/MEM magnitude amp = {np.round(FA.RBC2MEMmag_amp,3)} = {np.round(200*FA.RBC2MEMmag_amp/np.mean(FA.RBC2MEMmag[100:]),2)} %",fontsize=12)
        draw_figure(window['-WIGGLES-'].TKCanvas,plt.gcf())

    while True:
        event, values = window.read()
        #print("")
        #print(event)
        #print(values)
        if event == sg.WIN_CLOSED:
            break
## --------------- LOAD FILE --------------------------- ##
        elif event == ('LoadFile'):
            if values['twixfile'] == True:
                FA = FlipCal(twix_path=values['filepath'].replace('"',''))
            elif values['picklefile'] == True:
                FA = FlipCal(pickle_path=values['filepath'].replace('"',''))
            elif values['matlabfile'] == True:
                FA = FlipCal(matlab_path=values['filepath'].replace('"',''))
            elif values['ismrmrdfile'] == True:
                FA = FlipCal(ismrmrd_path=values['filepath'].replace('"',''))
            else:
                print('No radio button selected?...')
            try:
                updateData()
                updateDecay()
                updateDP()
            except:
                pass

## --------------- PROCESS --------------------------- ##
        elif event == ('process'):
            FA.process()
            print(f"\033[36mConsole printout for \033[32m{FA.patientInfo['PatientName']}\033[36m imaged at \033[32m{FA.scanParameters['scanDate']} \033[37m")
            print(f"\033[36mGas Frequency should be set to \033[32m{np.round(FA.newGasFrequency,0)}\033[37m")
            try:
                print(f"\033[36mSet the Ref Voltage to \033[32m{np.round(FA.newVoltage,0)}\033[37m")
            except:
                print('No ref voltage')
            print(f"\033[36mTE90 is \033[32m{np.round(FA.TE90,3)}\033[37m")
            updateData()
            updateDecay()
            updateDP()
## --------------- Process Wiggles BUTTONS --------------------------- ##
        elif event == ('wiggles'):
            try:
                updateWiggles()
            except:
                FA.fit_all_DP_FIDs(goFast=False)
                updateWiggles()
## --------------- SAVE PICKLE BUTTON --------------------------- ##
        elif event == ('savedata'):
            SAVEpath = os.path.join(values['SAVEpath'].replace('"',''),f"FlipCal_{FA.patientInfo['PatientName']}_{FA.scanParameters['scanDate']}/")
            if not os.path.isdir(SAVEpath):
                os.makedirs(SAVEpath)
            try:
                FA.pickleMe(pickle_path=os.path.join(SAVEpath,f"{FA.patientInfo['PatientName']}_{FA.scanParameters['scanDate']}.pkl"))
            except:
                print('Pickle save failed at GUI level')    
            try:
                FA.printout(save_path=os.path.join(SAVEpath,f"{FA.patientInfo['PatientName']}_{FA.scanParameters['scanDate']}.png"))
            except:
                print('Printout save failed at GUI level')    
            try:
                dummy_dicom_path = values['dummy_dicom_path'].replace('"','')
                FA.dicomPrintout(dummy_dicom_path= dummy_dicom_path,save_path=os.path.join(SAVEpath,f"FlipCal_DICOMS"))
            except:
                print('DICOM save failed at GUI level')    
## --------------- Info Edit Buttons ------------------- ##
        elif event == ('editPatientName'):
            text = sg.popup_get_text('Enter Subject ID: ',default_text=FA.patientInfo['PatientName'])
            window['subject'].update(f'Subject: {text}')
            FA.patientInfo['PatientName'] = text
        elif event == ('editStudyDate'):
            text = sg.popup_get_text('Enter Study Date: ',default_text=FA.scanParameters['scanDate'])
            window['date'].update(f'Date: {text}')
            FA.patientInfo['scanDate'] = text
        elif event == ('editDE'):
            text = sg.popup_get_text('Enter DE [mL]: ',default_text=FA.patientInfo['DE'])
            window['DE'].update(f'DE: {text}')
            FA.patientInfo['scanDate'] = text
