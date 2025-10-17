#Avantes spectrometers control library for BlickO.
#It is also a standalone library (together with spec_xfus) that can be used to control Avantes spectrometers
#directly (see the __main__ section at the end of the file).
#Written by Daniel Santana

from spec_xfus import spec_clock, calc_msl
import logging
import ctypes
from ctypes import windll,c_char,Structure,c_uint,c_byte,c_ushort,sizeof,byref,c_ubyte,c_float,c_uint8,c_uint16,c_uint32,c_double,c_bool,c_int
import numpy as np
from time import sleep
from copy import deepcopy
import sys
import os
import threading
import platform
#from matplotlib import pyplot as plt

if sys.version_info[0] < 3:
    # Python 2.x
    from Queue import Queue
else:
    # Python 3.x
    from queue import Queue




#create a module logger
logger=logging.getLogger(__name__)

#---Constants---
AVS_SERIAL_LEN = 10
USER_ID_LEN = 64

devtypes={0:"TYPE_UNKNOWN",
          1:"TYPE_AS5216",
          2:"TYPE_ASMINI",
          3:"TYPE_AS7010",
          4:"TYPE_AS7007"}

dev_status={0:"ETH_CONN_STATUS_CONNECTING",
            1:"ETH_CONN_STATUS_CONNECTED",
            2:"ETH_CONN_STATUS_CONNECTED_NOMON",
            3:"ETH_CONN_STATUS_NOCONNECTION",
            }

errors={ 0: "OK",
        -1: "Function called with invalid parameter value.",
        -2: "Operation not supported, ie: Function called to use 16bit ADC mode, with 14bit ADC hardware.",
        -3: "Opening communication failed or time-out during communication occurred.",
        -4: "AvsHandle is unknown in the DLL.",
        -5: "Function is called while result of previous function is not received yet.",
        -6: "No answer received from device.",
        -7: "Reserved (-7)",
        -8: "No measurement data is received at the point AVS_GetScopeData is called (Invalid meas data).",
        -9: "Allocated buffer size to small.",
        -10: "Measurement preparation failed because pixel range is invalid.",
        -11: "Measurement preparation failed because integration time is invalid (for selected sensor).",
        -12: "Measurement preparation failed because invalid combination of parameters, e.g. integration time of (600000) and (Navg >5000).",
        -13: "Reserved (-13)",
        -14: "Measurement preparation failed because no measurement buffers available.",
        -15: "Unknown error reason received from spectrometer.",
        -16: "Error in communication occurred.",
        -17: "No more spectra available in RAM, all read or measurement not started yet.",
        -18: "DLL version information can not be retrieved.",
        -19: "Memory allocation error in the DLL.",
        -20: "Function called before AVS_Init() is called.",
        -21: "Function failed because AvaSpec is in wrong state (e.g AVS_StartMeasurement while measurement is pending).",
        -22: "Reply is not a recognized protocol message",
        -24: "Error occurred while opening a bus device on the host. E.g. USB device access denied due to user rights",
        -25: "A read error has occurred when reading the onboard temperature sensor",
        -26: "A write error has occurred.",
        -27: "Library could not be initialized due to an Ethernet connection initialization error.",
        -28: "The device-type information stored in the spectrometer isn't recognized as one of the known device types.",
        -29: "The AVS_GetDeviceType function is used, but the secure config (holding the device type information) hasn't been read yet. Most likely the device isn't initialised correctly.",
        -30: "Unexpected response from spectrometer while getting measurement data",
        -100: "NrOfPixel in Device data incorrect.",
        -101: "Gain Setting Out of Range.",
        -102: "OffSet Setting Out of Range.",
        -120: "Use of AVS_SetSensitivityMode() not supported by detector type", #dll v9.11: this error might be given if we try to set the sensitivity mode to a detector that does not allow it
        -121: "Use of AVS_SetSensitivityMode() not supported by firmware version", #dll v9.11: this error might be given if we try to set the sensitivity mode to a detector that allows it but it has not the proper firmware version
        -122: "Use of AVS_SetSensitivityMode() not supported by FPGA version", #dll v9.11: this error might be given if we try to set the sensitivity mode to a detector that allows it but it has not the proper FPGA version
        -141: "Incorrect start pixel found in EEPROM",
        -142: "Incorrect end pixel found in EEPROM",
        -143: "Incorrect start or end pixel found in EEPROM",
        -144: "Factor should be in range 0.0 -4.0",
        -999: "Spectrometer operation timed out.",
        1000: "Invalid Handle.",
        }

#---Structures---
def create_AVS_classes():
    global AvsIdentityType
    global BroadcastAnswerType
    global MeasConfigType
    global DeviceConfigType
    global DstrStatusType

    class AvsIdentityType(ctypes.Structure):
      _pack_ = 1
      _fields_ = [("SerialNumber", ctypes.c_char * AVS_SERIAL_LEN),
                  ("UserFriendlyName", ctypes.c_char * USER_ID_LEN),
                  ("Status", ctypes.c_char)]

    class BroadcastAnswerType(ctypes.Structure):
      _pack_ = 1
      _fields_ = [("InterfaceType", ctypes.c_uint8),
                  ("serial", ctypes.c_char * AVS_SERIAL_LEN),
                  ("port", ctypes.c_uint16),
                  ("status", ctypes.c_uint8),
                  ("RemoteHostIp", ctypes.c_uint32),
                  ("LocalIp", ctypes.c_uint32),
                  ("reserved", ctypes.c_uint8 * 4)]

    class MeasConfigType(ctypes.Structure):
      _pack_ = 1
      _fields_ = [("m_StartPixel", ctypes.c_uint16), #First pixel to be sent to the pc
                  ("m_StopPixel", ctypes.c_uint16),
                  ("m_IntegrationTime", ctypes.c_float),
                  ("m_IntegrationDelay", ctypes.c_uint32),
                  ("m_NrAverages", ctypes.c_uint32),
                  ("m_CorDynDark_m_Enable", ctypes.c_uint8), # nesting of types does NOT work!!
                  ("m_CorDynDark_m_ForgetPercentage", ctypes.c_uint8),
                  ("m_Smoothing_m_SmoothPix", ctypes.c_uint16),
                  ("m_Smoothing_m_SmoothModel", ctypes.c_uint8),
                  ("m_SaturationDetection", ctypes.c_uint8),
                  ("m_Trigger_m_Mode", ctypes.c_uint8),
                  ("m_Trigger_m_Source", ctypes.c_uint8),
                  ("m_Trigger_m_SourceType", ctypes.c_uint8),
                  ("m_Control_m_StrobeControl", ctypes.c_uint16),
                  ("m_Control_m_LaserDelay", ctypes.c_uint32),
                  ("m_Control_m_LaserWidth", ctypes.c_uint32),
                  ("m_Control_m_LaserWaveLength", ctypes.c_float),
                  ("m_Control_m_StoreToRam", ctypes.c_uint16)]

    class DeviceConfigType(ctypes.Structure):
      _pack_ = 1
      _fields_ = [("m_Len", ctypes.c_uint16),
                  ("m_ConfigVersion", ctypes.c_uint16),
                  ("m_aUserFriendlyId", ctypes.c_char * USER_ID_LEN),
                  ("m_Detector_m_SensorType", ctypes.c_uint8),
                  ("m_Detector_m_NrPixels", ctypes.c_uint16),
                  ("m_Detector_m_aFit", ctypes.c_float * 5),
                  ("m_Detector_m_NLEnable", ctypes.c_bool),
                  ("m_Detector_m_aNLCorrect", ctypes.c_double * 8),
                  ("m_Detector_m_aLowNLCounts", ctypes.c_double),
                  ("m_Detector_m_aHighNLCounts", ctypes.c_double),
                  ("m_Detector_m_Gain", ctypes.c_float * 2),
                  ("m_Detector_m_Reserved", ctypes.c_float),
                  ("m_Detector_m_Offset", ctypes.c_float * 2),
                  ("m_Detector_m_ExtOffset", ctypes.c_float),
                  ("m_Detector_m_DefectivePixels", ctypes.c_uint16 * 30),
                  ("m_Irradiance_m_IntensityCalib_m_Smoothing_m_SmoothPix", ctypes.c_uint16),
                  ("m_Irradiance_m_IntensityCalib_m_Smoothing_m_SmoothModel", ctypes.c_uint8),
                  ("m_Irradiance_m_IntensityCalib_m_CalInttime", ctypes.c_float),
                  ("m_Irradiance_m_IntensityCalib_m_aCalibConvers", ctypes.c_float * 4096),
                  ("m_Irradiance_m_CalibrationType", ctypes.c_uint8),
                  ("m_Irradiance_m_FiberDiameter", ctypes.c_uint32),
                  ("m_Reflectance_m_Smoothing_m_SmoothPix", ctypes.c_uint16),
                  ("m_Reflectance_m_Smoothing_m_SmoothModel", ctypes.c_uint8),
                  ("m_Reflectance_m_CalInttime", ctypes.c_float),
                  ("m_Reflectance_m_aCalibConvers", ctypes.c_float * 4096),
                  ("m_SpectrumCorrect", ctypes.c_float * 4096),
                  ("m_StandAlone_m_Enable", ctypes.c_bool),
                  ("m_StandAlone_m_Meas_m_StartPixel", ctypes.c_uint16),
                  ("m_StandAlone_m_Meas_m_StopPixel", ctypes.c_uint16),
                  ("m_StandAlone_m_Meas_m_IntegrationTime", ctypes.c_float),
                  ("m_StandAlone_m_Meas_m_IntegrationDelay", ctypes.c_uint32),
                  ("m_StandAlone_m_Meas_m_NrAverages", ctypes.c_uint32),
                  ("m_StandAlone_m_Meas_m_CorDynDark_m_Enable", ctypes.c_uint8),
                  ("m_StandAlone_m_Meas_m_CorDynDark_m_ForgetPercentage", ctypes.c_uint8),
                  ("m_StandAlone_m_Meas_m_Smoothing_m_SmoothPix", ctypes.c_uint16),
                  ("m_StandAlone_m_Meas_m_Smoothing_m_SmoothModel", ctypes.c_uint8),
                  ("m_StandAlone_m_Meas_m_SaturationDetection", ctypes.c_uint8),
                  ("m_StandAlone_m_Meas_m_Trigger_m_Mode", ctypes.c_uint8),
                  ("m_StandAlone_m_Meas_m_Trigger_m_Source", ctypes.c_uint8),
                  ("m_StandAlone_m_Meas_m_Trigger_m_SourceType", ctypes.c_uint8),
                  ("m_StandAlone_m_Meas_m_Control_m_StrobeControl", ctypes.c_uint16),
                  ("m_StandAlone_m_Meas_m_Control_m_LaserDelay", ctypes.c_uint32),
                  ("m_StandAlone_m_Meas_m_Control_m_LaserWidth", ctypes.c_uint32),
                  ("m_StandAlone_m_Meas_m_Control_m_LaserWaveLength", ctypes.c_float),
                  ("m_StandAlone_m_Meas_m_Control_m_StoreToRam", ctypes.c_uint16),
                  ("m_StandAlone_m_Nmsr", ctypes.c_int16),
                  ("m_DynamicStorage", ctypes.c_uint8 * 12), # ex SD Card, do not use
                  ("m_Temperature_1_m_aFit", ctypes.c_float * 5),
                  ("m_Temperature_2_m_aFit", ctypes.c_float * 5),
                  ("m_Temperature_3_m_aFit", ctypes.c_float * 5),
                  ("m_TecControl_m_Enable", ctypes.c_bool),
                  ("m_TecControl_m_Setpoint", ctypes.c_float),
                  ("m_TecControl_m_aFit", ctypes.c_float * 2),
                  ("m_ProcessControl_m_AnalogLow", ctypes.c_float * 2),
                  ("m_ProcessControl_m_AnalogHigh", ctypes.c_float * 2),
                  ("m_ProcessControl_m_DigitalLow", ctypes.c_float * 10),
                  ("m_ProcessControl_m_DigitalHigh", ctypes.c_float * 10),
                  ("m_EthernetSettings_m_IpAddr", ctypes.c_uint32),
                  ("m_EthernetSettings_m_NetMask", ctypes.c_uint32),
                  ("m_EthernetSettings_m_Gateway", ctypes.c_uint32),
                  ("m_EthernetSettings_m_DhcpEnabled", ctypes.c_uint8),
                  ("m_EthernetSettings_m_TcpPort", ctypes.c_uint16),
                  ("m_EthernetSettings_m_LinkStatus", ctypes.c_uint8),
                  ("m_EthernetSettings_m_ClientIdType", ctypes.c_uint8),
                  ("m_EthernetSettings_m_ClientIdCustom", ctypes.c_char * 32),
                  ("m_EthernetSettings_m_Reserved", ctypes.c_uint8 * 79),
                  ("m_Reserved", ctypes.c_uint8 * 9608),
                  ("m_OemData", ctypes.c_uint8 * 4096)]

    class DstrStatusType(ctypes.Structure):
      _pack_ = 1
      _fields_ = [("m_TotalScans", ctypes.c_uint32),
                  ("m_UsedScans", ctypes.c_uint32),
                  ("m_Flags", ctypes.c_uint32),
                  ("m_IsStopEvent", ctypes.c_uint8),
                  ("m_IsOverflowEvent", ctypes.c_uint8),
                  ("m_IsInternalErrorEvent", ctypes.c_uint8),
                  ("m_Reserved", ctypes.c_uint8)]
create_AVS_classes()

#Dictionary to store the memory addresses of the Avantes_Spectrometers instantiated objects, indexed by their spec_id.
#Needed by measure_callback()
Avantes_Spectrometer_Instances={}
def measure_callback(pparam1, pparam2):
    """
    Measure callback function (global function).
    The Avantes dll will call to this global function once a measurement has been finished,
    in order to notify that the measurement data is ready to be queried.
    Note: due to restrictions in the AVS_MeasureCallback() function, this must be a global function.

    params:
     <pparam1>: (ldev_handle) spectrometer id (The same spec_id given by AVS_Activate). (integer)
     <pparam2>: (lerror) nmeas or error code (integer);
      - 0 means no error.
      - >0 means the number of cycles available in device RAM memory to be read (if StoreToRAM parameter > 0)
      - or INVALID_MEAS_DATA = -8

    """
    arrival_time=spec_clock.now()
    #deference the pointers.
    ldev_handle = pparam1[0] #device handle (spec_id)
    lerror = pparam2[0] #nmeas or error
    #logger.debug("Meaurement Callback!, ldev="+str(ldev_handle)+", lerr="+str(lerror))

    if ldev_handle in Avantes_Spectrometer_Instances:
        #There will be one Avantes_Spectrometer() instance per spectrometer connected to the computer.
        #At connection time, the instance is added to the Avantes_Spectrometer_Instances dictionary, indexed by its spec_id.
        #here we get the instance of the spec that called this callback function.
        instance=Avantes_Spectrometer_Instances[ldev_handle]

        #Put a tuple with the error code and the data arrival time into the get data queue,
        #to notify to the data arrival watchdog thread that a measurement is done and data is ready to be read.
        if instance.docatch:
            if lerror<0: #Error
                if instance.debug_mode>2:
                    instance.logger.debug("measure_callback, spec "+instance.alias+" reported error code "+str(lerror)+".")
                instance.read_data_queue.put((lerror,arrival_time))
            elif lerror==0: # 1 Measurement available, when StoreToRam is disabled:
                if instance.debug_mode>2:
                    instance.logger.debug("measure_callback, spec "+instance.alias+" has 1 measurement ready to be read.")
                instance.read_data_queue.put((lerror,arrival_time))
            else: #X measurements available, when StoreToRam is enabled:
                if instance.debug_mode>2:
                    instance.logger.debug("measure_callback, spec "+instance.alias+" has "+str(lerror)+" measurements ready to be read.")
                for i in range(lerror):
                    instance.read_data_queue.put((0,arrival_time))

    return

def connection_status_callback(pparam1, pparam2):
    """
    Connection status changed callback function.
    Note: only valid for non USB spectrometers.
    """
    # deference the pointers.
    ldev_handle = pparam1[0] #device handler (spec_id)
    cstat = pparam2[0] #connection status
    logger.warning("Connection status Callback!, ldev="+str(ldev_handle)+", cstat="+str(cstat))

    if ldev_handle in Avantes_Spectrometer_Instances:

        instance=Avantes_Spectrometer_Instances[ldev_handle]
        if cstat in dev_status:
            logger.info("Connection status of spec "+instance.alias+" changed to: "+dev_status[cstat])
    return

class Avantes_Spectrometer():

    def __init__(self):
        #Note: Next to the parameters description there will be an (E) or an (I).
        # (E) means that the parameter may be eventually accessed / configured by BlickO.
        # (I) means that the parameter is for internal usage only, and won't be accessed by BlickO.

        #---Parameters---
        self.debug_mode=0 #(E) 0=disabled, >0=enabled. Low numbers are less verbose, high numbers are more verbose.
        #debug_mode=0 will disable the debug mode, only the basic connection/disconnection steps (with self.logger.info()) will be printed in the log files.
        #debug_mode=1 will print the previous plus the debug messages related to start and end of the measurements.
        #debug_mode=2 will print the previous plus the changes of integration time (set_it() function).
        #debug_mode=3 will print the previous plus reset_data, read_data and handle_data actions (every cycle actions).


        self.simulation_mode=False #(E) Set this to True to enable the simulation mode (no dll will be loaded, no spectrometer will be really connected)
        self.dll_logging=False #(E) Set this to True to enable the internal logging of the dll. (Only for debugging purposes)

        #Avantes control DLL path
        self.dll_path= os.path.abspath("../../lib/oslib/spec_ava1/Avaspec-DLL_9.14.0.9_64bits/ava.dll") #(E) Will store the path of the Avantes control dll (string)
        #Spec Parameters:
        self.sn= "2203162U1" #"1102185U1" #(E) Serial number of the spectrometer to be used (string)
        self.alias="1" #(E) Alias of the spectrometer (string, just to identify the spec in the log files)
        self.npix_active=2048 #(E) Number of active pixels of the detector (integer)
        self.npix_blind_left=0 #(E) Number of blind pixels at left side of the detector (integer)
        # Note: the total number of pixels will be = npix_blind_left + npix_active
        self.nbits=16 #(E) Number of bits of the spectrometer detector A/D converter (integer)
        self.max_it_ms=4000.0 #(E) Maximum integration time allowed by the spectrometer, in milliseconds (float) (=Max exposure time)
        self.min_it_ms=2.4 #(E) Minimum integration time allowed by the spectrometer, in milliseconds (float) (=Min exposure time)
        self.discriminator_factor=4.0 #(E) Will store the discriminator factor to be applied to the counts (float)
        self.sensitivity_mode=-1 #(E) -1 = do not set, 0=low noise, 1=high sensitivity. (Only valid for specific detectors, ie: paNIR)
        self.eff_saturation_limit=2**self.nbits-1 #(E) Effective saturation limit of the detector [max counts] (integer). Measurements containing counts above this limit will be considered as saturated.

        #Working mode:
        self.abort_on_saturation=True #(E) boolean - If True, the measurement will be aborted as soon as saturated signal
        # is detected in the latest cycle read data. An order to abort the rest of the measurement will be sent to the spectrometer
        # and no more data will be handled from that moment. The output data would be still usable, but it would only
        # contain the non-saturated cycles (if any).

        self.store_to_ram=False #(E) boolean - Experimental feature of Avantes. Set this to True to enable
        # the StoreToRam option of the Avantes spectrometer.
        # When this option is enabled, the working mode of the spectrometer is the following:
        # measure -> store in roe RAM -> flag to pc (and start next measurement asap) -> pc reads data from roe RAM.
        # When this option is disabled (default), the working mode is:
        # measure -> send data to the dll input buffer -> flag to pc -> pc reads data from the dll input buffer ->
        # the spectrometer starts the next measurement.
        # (note that some spectrometers are able to measure while transferring the previous data, and others not)
        # When this option is enabled, we get more evenly spaced data arrival times. But this option is not
        # compatible with blind pixels. So only can be used for spectrometers without blind pixels.
        # Also, the number of measurements that can be stored in the ROE RAM is limited, depending on the roe RAM

        #Performance tests:
        self.performance_test_it_ms_list=np.arange(2.4,10.1,0.1) #(I) List or Array with the different integration times to be tested during the performance test. [ms]
        self.performance_test_ncy_list=[1,10,100,500,1000,2000] #(I) List or Array with the different number of cycles to be tested during the performance test. [list of int]

        #--------Do not modify anything below this line, the following variables are for internal usage only----
        self.dll_handler=None #(E) Will store the handler to the Avantes control dll (ctypes.CDLL object)
        self.spec_id=None #(E) Will store the spectrometer id, used by some dll functions in order to point to one specific spectrometer device (byte string)
        self.parlist=None #(E) Will be used to store the low level parameter list (internal configuration parameters of the spectrometer).
        self.it_ms=None #(E) Will store the currently set integration time in milliseconds. Use set_it() to change it.
        self.logger=None #(E) Will store the logger object for one specific spectrometer (logging.Logger object, see initialize_spec_logger())
        self.product_id=None #(I) Will store the spectrometer product id, used by the dll to initialize itself for a specific spectrometer model
        self.devtype=None #(I) Will store the spectrometer device type (ROE type, ie AS5216 or AS7010) (string). Used for internal recovery protocols.

        #Internal variables for measure control
        self.measuring=False #(E) Will be used to indicate that the spectrometer is measuring (boolean)
        self.recovering=False #(E) Will be used to indicate that the spectrometer is in recovery mode (boolean) -> it will disable the external_meas_done_event
        self.docatch=False #(E) Will be used to "hear" for data arrival events. If False, the data arrival events will be ignored.
        self.busy=False #(E) Will be used to know when the spectrometer dll is busy (boolean) -> currently only used in read_data, in order not to send an stop measure order (due saturation) while the dll is reading data.
        self.ncy_requested=0 #(E)Will store the number of measurements requested (integer)
        self.ncy_read=0 #(E) Will store the current number of measurements done (integer)
        self.ncy_saturated=0 #(E) Will store how many saturated measurements there are in the handled data (integer)
        self.internal_meas_done_event=threading.Event() #(I) This internal event will be "unset" whenever a measurement is started, and "set" when the measurement is complete (all ncy read and handled). Its usage is internal: just for this module.

        #Internal variables for data output:
        self.rcm=np.array([]) #(E) Will store the mean raw counts of the measurements (numpy array)
        self.rcs=np.array([]) #(E) Will store the (sample) standard deviation of the raw counts of the measurements (numpy array)
        self.rcl=np.array([]) #(E) Will store the rms of the standard deviation fitted to a straight line
        self.rcm_blind_left=np.array([]) #(E) same as rcm, but for the blind pixels at the left side of the detector
        self.rcs_blind_left=np.array([]) #(E) same as rcs, but for the blind pixels at the left side of the detector
        self.rcl_blind_left=np.array([]) #(E) same as rcl, but for the blind pixels at the left side of the detector

        #Post-processing actions
        self.external_meas_done_event=None #(E) External event to be set when a measurement is complete (apart from the internal_meas_done_event). (None or threading.Event object, Optional).
        #This external event will be used to notify to other parent modules that a measurement is complete.
        #Note: This external event must be unset by the parent module, this module will not unset it.

        #Internal variables to get data and handle data (queues and threads):
        self.read_data_queue=Queue() #(I) Will store the get data queue. When a measurement is ready, a flag will be put here by measure_callback().
        self.handle_data_queue=Queue() #(I) Will store the data arrival queue. When a measurement is done, the data will be put here for subsequent data handling.
        self.data_arrival_watchdog_thread=None #(I) Will store the data arrival watchdog thread
        self.data_handling_watchdog_thread=None #(I) Will store the data handling watchdog thread

        #(I) Measurement callback function: avantes dll will call this function to notify that a measurement is ready to be read.
        CALLBACK_FUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        self.measure_cb_function = CALLBACK_FUNC_TYPE(measure_callback)

        #(I) Connection status callback function: avantes dll will call this function to notify that the connection status has changed.
        CALLBACK_FUNC_TYPE = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        self.conn_status_cb_function = CALLBACK_FUNC_TYPE(connection_status_callback) #Note -> not working for USB spectrometers.

        #(I) For simulation mode:
        self.simulated_measurement_timer=None #Will store a timer that will emulate data arrivals (threading.Timer object)

        #Error Handling

        #(I) Database of possible dll error codes:
        self.errors = errors

        self.error="OK" #(E) Latest generated error description (string). See self.errors for a list of possible error descriptions.
        # It can also be "OK" if no error, or "UNKNOWN_ERROR" if the dll returned an error code that is not in the list of known errors.

        self.last_errcode=0 #(I) Latest generated dll error code (integer). See self.errors for a list of possible error codes.


        #----------------------------------------------

    #---Main Control Functions (used by BlickO)---

    def initialize_spec_logger(self):
        self.logger = logging.getLogger("spec"+self.alias)

    def connect(self):
        """
        Connects to the spectrometer and initializes it.
        """

        res="OK"

        #Reset data:
        self.reset_spec_data()

        if self.simulation_mode:
            self.logger.info("--- Connecting spectrometer "+self.alias+"... (Simulation Mode ON) ---")
            self.spec_id=len(Avantes_Spectrometer_Instances)+1 #Create a new spec_id
            Avantes_Spectrometer_Instances[self.spec_id]=self #Add this class instance to the instances dictionary, indexed by its spec_id.

        else:
            self.logger.info("--- Connecting spectrometer "+self.alias+"... ---")

            #Load the spectrometer control dll:
            res=self.load_spec_dll() #This will be only needed at initial connection.

            #Initialize dll
            if res=="OK":
                res=self.initialize_dll() #AVS_Init()

            #Enable/disable dll debug mode:
            if res=="OK":
                res=self.enable_dll_logging(enable=self.dll_logging)

            #Check number of USB connected devices is >0
            if res=="OK":
                res,ndev=self.get_number_of_devices() #AVS_UpdateUSBDevices()

            #Get device information of all connected specs
            if res=="OK":
                res,l_pData_all=self.get_all_devices_info(ndev)

            #Find spectrometer with correct serial number:
            if res=="OK":
                res,l_pData=self.find_spec_info(ndev,l_pData_all)
                #print(l_pData)

            #Activate spectrometer and get device id:
            if res=="OK":
                res,self.spec_id=self.activate_spec(l_pData)
                sleep(0.2)

            #Add device id to the Avantes_Spectrometer_Instances dictionary, indexed by its spec_id:
            if res=="OK":
                Avantes_Spectrometer_Instances[self.spec_id]=self
                #Add this class instance to the instances dictionary, indexed by its alias.
                #This will be used by the measure_callback() function to get the spec data once it is ready to be read.

            #Register change-of-status callback function:
            # if res=="OK":
            #     res=self.register_status_callback() #Not used for USB spectrometers.

            if res=="OK":
                _=self.get_device_config() #This step is not essential, skip res if not successful

            #Print device info, ROE model:
            if res=="OK":
                _,self.devtype=self.get_device_type() #This step is not essential, skip res if not successful

            #Print device info: Detector model:
            if res=="OK":
                _,_=self.get_detector_name() #This step is not essential, skip res if not successful

            #Print device info: FPGA version, firmware version, and dll version:
            if res=="OK":
                _,_,_,_=self.get_version_info() #This step is not essential, skip res if not successful

            #Send an order to stop any ongoing measurement:
            if res=="OK":
                res=self.abort(ignore_errors=True)

            #Get number of <active> pixels:
            if res=="OK":
                res,npix=self.get_number_of_pixels()

            #Check number of pixels is correct:
            if res=="OK":
                if npix!=self.npix_active:
                    res="Number of active pixels of spec "+self.alias+" is "+str(npix)+", expected "+str(self.npix_active)+". Check IOF parameters."

            #Check number of blind pixels is correct:
            #There is no function to check this... so we will skip this step.

            #Set Sensitivity mode (only if self.sensitivity_mode!=-1):
            if res=="OK":
                if self.sensitivity_mode!=-1:
                    res=self.set_sensitivity(self.sensitivity_mode)

            #Initialize structures and set fixed parameters:
            if res=="OK":
                self.parlist=MeasConfigType()
                self.parlist.m_StartPixel=c_uint16(0) #First pixel to be sent to the pc
                self.parlist.m_StopPixel=c_uint16(npix-1) #Last pixel to be sent to the pc
                l_NanoSec = -21
                l_FPGAClkCycles = int(round(6.0*(l_NanoSec+20.84)/125.0))
                self.parlist.m_IntegrationDelay=c_uint32(l_FPGAClkCycles)
                self.parlist.m_NrAverages=c_uint32(1) #Number of averages in a single measurement
                self.parlist.m_CorDynDark_m_Enable=c_uint8(0)
                self.parlist.m_CorDynDark_m_ForgetPercentage=c_uint8(0)
                self.parlist.m_Smoothing_m_SmoothPix=c_uint16(0)
                self.parlist.m_Smoothing_m_SmoothModel=c_uint8(0)
                self.parlist.m_SaturationDetection=c_uint8(0)
                self.parlist.m_Trigger_m_Mode=c_uint8(0)
                self.parlist.m_Trigger_m_Source=c_uint8(0)
                self.parlist.m_Trigger_m_SourceType=c_uint8(0)
                self.parlist.m_Control_m_StrobeControl=c_uint16(0)
                self.parlist.m_Control_m_LaserDelay=c_uint32(0)
                self.parlist.m_Control_m_LaserWidth=c_uint32(0)
                self.parlist.m_Control_m_LaserWaveLength=c_float(0)
                if self.store_to_ram:
                    self.logger.info("StoreToRam option is enabled.")
                    self.parlist.m_Control_m_StoreToRam=c_uint16(1)
                else:
                    self.logger.info("Store to (ROE) RAM option is disabled.")
                    self.parlist.m_Control_m_StoreToRam=c_uint16(0)

        #Set initial integration time:
        if res=="OK":
            #set integration time to the double of the minimum and then the minimum
            #to check whether integration time change works
            for it in [self.min_it_ms * 2,self.min_it_ms]:
                res=self.set_it(it)
                if res!="OK":
                    break
            sleep(0.2)

        #Create a side threads with the "data arrival watchdog" and the "data handling watchdog":
        if res=="OK":
            if self.data_arrival_watchdog_thread is None:
                self.logger.info("Starting data arrival watchdog thread for spectrometer "+self.alias+".")
                self.data_arrival_watchdog_thread=threading.Thread(target=self.data_arrival_watchdog)
                self.data_arrival_watchdog_thread.start()
            if self.data_handling_watchdog_thread is None:
                self.logger.info("Starting data handling watchdog thread for spectrometer "+self.alias+".")
                self.data_handling_watchdog_thread=threading.Thread(target=self.data_handling_watchdog)
                self.data_handling_watchdog_thread.start()

        return res

    def set_it(self,it_ms):
        """
        Set Avantes spectrometer integration time (=Exposure time)

        params:
            <it_ms>: Integration time in milliseconds (float)
        """
        res="OK"
        if self.simulation_mode:
            if self.debug_mode>=2:
                self.logger.debug("Setting integration time to "+str(it_ms)+" ms (Simulation mode)")
            self.it_ms=it_ms
        else:
            #Ensure it is a float:
            it_ms=float(it_ms) #convert to float (possibly integer to double precision float)
            c_it_ms=c_float(it_ms) #convert to c_float, which is what the m_IntegrationTime field needs.
            # Note -> Here we are converting a double precision float to a single precision float,
            # therefore there might be a loss of precision.
            if self.debug_mode>=2:
                self.logger.debug("Setting integration time to "+str(it_ms)+" ms, ("+str(c_it_ms.value)+" ms)")
            #Update parameters list:
            self.parlist.m_IntegrationTime=c_it_ms
            #Call AVS_PrepareMeasure to update the new config parameters:
            resdll=self.dll_handler.AVS_PrepareMeasure(self.spec_id,byref(self.parlist))
            res=self.get_error(resdll)
            if res=="OK":
                #TODO: Check the IT that was really set. (Cannot find a function to query the current IT)
                #Update currently configured integration time:
                self.it_ms=it_ms
            else:
                res="Could not set integration time to "+str(it_ms)+" ms. Error: "+res
                self.logger.error(res)

        self.error=res
        return res

    def measure(self,ncy=1):
        """
        res=measure(ncy=10)
        For BlickO integration, it MUST be a non blocking call.

        This function will request to the spectrometer to start ncy measurements.
        The spec dll will call to the callback function for every finished ncy.
        The callback function will add a flag into the read_data_queue() to indicate
        to the data arrival watchdog thread that a measurement is ready to be read.
        The data arrival watchdog thread will notice this flag and will proceed to read the spectrometer data
        (by using self.read_data). Then the data read is put into the handle_data_queue()
        to indicate to the data handling watchdog thread that new data is ready to be handled.
        The data handling watchdog thread will notice this new data flag and will proceed to handle the data.

        Finally, once all cycles are measured, read, and handled, the data handling watchdog thread will calculate
        the mean and standard deviation, and will set the flag self.measuring to False. If an external event is set,
        the data handling watchdog thread will also set this external event, to notify to the parent module that
        the measurement is complete.

        This can be much simpler, but in this way we ensure the spec is already measuring the next cycle while
        we are handling the data of the previous cycle.

        params:
            <ncy>: Number of cycles to measure (integer)
            <abort_on_saturation>: (boolean) Set this to True to abort the measurement if saturation is detected.
             (only possible if store_to_ram is disabled)

        """
        self.internal_meas_done_event.clear()
        self.docatch=True
        self.measuring=True

        self.ncy_requested=ncy
        self.reset_spec_data() #reset accumulated data and measurements done/handled counters

        if self.simulation_mode:
            if self.debug_mode>=1:
                self.logger.debug("Starting measurement of spectrometer "+self.alias+", ncy="+str(ncy)+", IT="+str(self.it_ms)+" ms (Simulation mode ON)")
            res="OK"
            if self.store_to_ram: #Create a timer to simulate the total (all cycles) measurement time:
                self.simulated_measurement_timer=threading.Timer(self.it_ms*1e-3*ncy,measure_callback,args=((self.spec_id,),(ncy,)))
            else: #Create a timer to simulate one cycle measurement time:
                self.simulated_measurement_timer=threading.Timer(self.it_ms*1e-3,measure_callback,args=((self.spec_id,),(0,)))
                #(data_arrival_watchdog function will create the following ones)
            self.meas_start_time=spec_clock.now() #Set measurement start time
            self.simulated_measurement_timer.start()
        else:
            if self.debug_mode>=1:
                self.logger.debug("Starting measurement of spectrometer "+self.alias+", ncy="+str(ncy)+", IT="+str(self.it_ms)+" ms...")
            #Send order to the spectrometer to measure ncy spectra:
            #AVS_MeasureCallback(Device identifier, Callback function, number of cycles to measure)
            #It is a non-blocking call function
            if self.store_to_ram:
                self.set_store_to_ram_ncy(ncy) #Set the number of cycles to store to RAM, when store_to_ram is enabled
                self.meas_start_time=spec_clock.now() #Set measurement start time
                resdll=self.dll_handler.AVS_MeasureCallback(self.spec_id,self.measure_cb_function,c_uint16(1)) #ncy must be 1 here when store_to_ram is enabled
            else:
                self.meas_start_time=spec_clock.now() #Set measurement start time
                resdll=self.dll_handler.AVS_MeasureCallback(self.spec_id,self.measure_cb_function,c_uint16(ncy))
            res=self.get_error(resdll)
            if res!="OK":
                res="Could not start measurement of spec "+self.alias+". Error: "+res
                self.logger.error("measure, "+res)
                self.measuring=False

        self.error=res
        return res

    def abort(self,ignore_errors=False):
        """
        res=self.abort(ignore_errors=False)

        Send an order to the spectrometer in order to stop any ongoing measurement.

        params:
         <ignore_errors>: if True, ignore any error that may happen while stopping the measurement (boolean)

        return:
            <res>: string with the result of the operation. It can be "OK" or an error description.
        """
        self.docatch=False # -> Stop adding new data arrivals into the read data queue.
        res="OK"
        if self.simulation_mode:
            self.logger.info("stop_measure, Stopping any spec ongoing measurement (simulation mode)...")
            pass
        else:
            self.logger.info("stop_measure, Stopping any spec ongoing measurement...")
            try:
                resdll=self.dll_handler.AVS_StopMeasure(self.spec_id)
                res=self.get_error(resdll)
                if res!="OK": #Wrong answer
                    res="Could not stop any ongoing measurement. Error: "+res
                    if ignore_errors: #Just warn, but return OK
                        self.logger.warning("abort, "+res)
                        res="OK"
                    else:
                        self.logger.error("abort, "+res)
            except Exception as e:
                    res="Exception happened while stopping any ongoing measurement: "+str(e)
                    if ignore_errors: #Just warn, but return OK:
                        self.logger.warning(res)
                        res="OK"
                    else:
                        self.logger.exception(e)

        self.measuring=False

        self.error=res

        return res

    def read_aux_sensor(self,sname="detector"):
        """
        Read an auxiliary sensor of the spectrometer. (Temperature, Humidity, etc)

        Note: at the moment all Avantes systems have the same conversion polynomials
        (separately for the detector and the board temp).
        If this changes, then the polynomials need to be added into the IOF or read from the EEProm
        """
        res="OK"
        value=-99.0
        if self.simulation_mode:
            value=11.1
        else:
            if sname =="detector":
                analogid=c_ubyte(0)
                vcut=5.0
                tc=[58.7,-20.48] #conversion polynomial to convert voltage -> temperature
            elif sname=="board_analog":
                analogid=c_ubyte(6)
                vcut=5.0
                tc=[118.69,-70.361,21.02,-3.6443,0.1993] #conversion polynomial to convert voltage -> temperature
            elif sname=="board_digital":
                #Specific avantes spectrometer models (AS7007 and AS7010) have digital sensors,
                #that returns the temperatures in degrees already.
                analogid=c_ubyte(6)
                vcut=99.0
                tc=[0.,1.] #digital reader, no conversion needed
            else:
                res="Unknown sensor name: '"+sname+"' for spec "+self.alias
                self.logger.error(res)

        if res=="OK":
            vc=c_float()
            resdll=self.dll_handler.AVS_GetAnalogIn(self.spec_id,analogid,byref(vc))
            res=self.get_error(resdll)
            if res=="OK":
                v=vc.value
                if v<vcut: #There is a sensor
                    value=np.polyval(tc[::-1],v) #Apply conversion polynomial
                    if self.debug_mode>=2:
                        self.logger.debug("read_aux_sensor, '"+sname+"' sensor value: "+str(round(value,4)))
                else:
                    self.logger.warning("read_aux_sensor, '"+sname+"' sensor value is out of range: "+str(v))
            else:
                self.logger.warning("read_aux_sensor, could not read '"+sname+"' aux sensor, err= "+res)

        return res,value

    def disconnect(self,dofree=False,ignore_errors=False,force_dofree=False):
        """
        Deactivate and disconnect spectrometer from dll.
        params:
            <dofree>: (boolean) Set this to True to finalize the dll communication as well (clear dll_handler).
            Note that this parameter will only have effect when there is only one spectrometer connected
            (ie when all the other spectrometers have been already disconnected). Otherwise, the dll communication
             won't be closed, because the other spectrometers would become unmanageable.
             To skip this safety check (i.e. at recovery procedures), you could set <force_dofree> to True.

            <ignore_errors>: (boolean) Set this to True to ignore any error that happens during the disconnection
             process.

        """
        res="OK"

        #Remove the instance from the Avantes_Spectrometer_Instances dictionary: So that no more data is read from this spectrometer.
        if self.spec_id in Avantes_Spectrometer_Instances:
            del Avantes_Spectrometer_Instances[self.spec_id]

        if self.simulation_mode:
            self.logger.info("Disconnecting spectrometer "+self.alias+"... (Simulation mode)")
        else:
            self.logger.info("Disconnecting spectrometer "+self.alias+"...")
            if self.dll_handler is not None:
                #Deactivate spec: Closes communication with selected spectrometer (clear self.spec_id)
                res=self.deactivate(ignore_errors=ignore_errors)

                #Finalize the dll communication (call AVS_Done() to clear the dll_handler):
                if dofree:
                    res=self.close_spec_dll(ignore_errors=ignore_errors, force_dofree=force_dofree)

                if ignore_errors:
                    res="OK"

        if self.data_arrival_watchdog_thread is not None:
            self.logger.info("Closing data arrival watchdog thread of spectrometer "+self.alias+".")
            self.read_data_queue.put((None,None)) #Send a "stop" signal to the data handling watchdog thread.
            self.data_arrival_watchdog_thread.join() #Wait for the data arrival watchdog thread to finish.
            self.data_arrival_watchdog_thread=None
        if self.data_handling_watchdog_thread is not None:
            self.logger.info("Closing data handling watchdog thread of spectrometer "+self.alias+".")
            self.handle_data_queue.put((None,None,(None,None))) #Send a "stop" signal to the data arrival watchdog thread.
            self.data_handling_watchdog_thread.join() #Wait for the data handling watchdog thread to finish.
            self.data_handling_watchdog_thread=None

        self.logger.info("Spectrometer "+self.alias+" disconnected.")

        self.error=res
        return res

    def recovery(self,ntry=3,dofree=False):
        """
        res=recovery(ntry=3)

        "Soft recovery" or "smooth recovery" of the spectrometer, by sending software commands.
        This function will be used to recover the control of a spectrometer that is not responding by either,
        too long integration time measurement, measurement data never arriving, or
        unexpected spectrometer power loss.

        It just recovers the control, but it does not to measure again the last interrupted
        measurement automatically.

        params:
            <ntry>: number of soft recovery cycles allowed, to try to recover the spectrometer. (integer)
            <dofree>: (boolean) is True, it will finalize the dll communication with AVS_Done() and
             (clear the dll_handler) as a first step. Note that this affects to ALL connected avantes spectrometers,
             so ensure all other spectrometers are iddle (not measuring).

        """
        self.docatch=False # -> Stop adding new data arrivals into the read data queue.

        for i in range(ntry):

            self.logger.warning("Recovering spectrometer "+self.alias+"... (try "+str(i+1)+"/"+str(ntry)+"), dofree="+str(dofree))

            if i==0 and dofree:
                res="NOK"
            else:
                #Try to stop any ongoing measurement:
                res=self.abort(ignore_errors=False)
                sleep(0.5)

            if res=="OK":
                #The spectrometer was able to abort the ongoing measurement: Then communication is still ok.
                self.logger.info("Spectrometer "+self.alias+" could abort any ongoing measurement.")

                #Set the last integration time set:
                res=self.set_it(it_ms=self.it_ms)
                sleep(0.5) #wait a bit for the integration time to be set.

                if res=="OK":
                    #the spectrometer was able to set the last integration time (and config) successfully:
                    self.logger.info("Spectrometer "+self.alias+" could set the last integration time. -> soft recovery finished successfully.")
                    break #Quit ntry for loop and exit

            if res!="OK":
                #If the code reach this point, the most possible reason is that the spectrometer lost its power while
                #it was measuring and the old dll handler is not valid anymore. So it is needed to finalize the current
                #dll instance, re-initialize the dll and re-connect the spec.

                #Try to disconnect the spectrometer, and finish the data arrival and data handling watchdog threads:
                dofree_now=True if i==0 and dofree else False
                _=self.disconnect(dofree=dofree_now, force_dofree=dofree_now) #Note: we need to use dofree otherwise it won't work.
                # But this has a side effect:
                #it will affect to all connected spectrometers.
                sleep(2) #wait a bit

                #Try a software hard-reset: This only works in the newest spectrometers (AS7010, AS7007 ROE, but not in AS5216)
                if self.devtype in ["TYPE_AS7010","TYPE_AS7007"]:
                    res=self.reset_device()

                #Try to connect the spectrometer again:
                res=self.connect()

                if res=="OK":
                    self.logger.info("Spectrometer "+self.alias+" recovered after stopping the last measurement, disconnecting, and re-connecting.")
                    break

                else:
                    self.logger.warning("Recovery of spectrometer "+self.alias+" failed.")
                    sleep(5)

        return res

    #---Main Control Functions (used by this library)---

    def measure_blocking(self,ncy=10):
        """
        Same as measure but blocking call
        """
        res=self.measure(ncy=ncy)
        if res=="OK":
            return self.wait_for_measurement()
        else:
            return res

    def wait_for_measurement(self):
        """
        res=wait_for_measurement()
        Wait for the measurement to be complete
        returns:
            <res>: If any problem happened while measuring, the last produced error will be stored in self.error
        """
        self.internal_meas_done_event.wait()
        return self.error


    #---Auxiliary functions (used by the Main functions)---

    #Auxiliary functions related to connection:

    def get_error(self,resdll):
        """
        res=get_error(resdll)
        Get the error description of an unsuccessful dll return answer.

        params:
            <resdll>: returned answer from a dll call (integer)

        return:
            <res>: error description (string)
        """
        #Save last generated error code (dll answer):
        self.last_errcode=resdll

        #Check dll answer meaning
        if resdll in self.errors:
            if resdll==0: #In case of no error, just return "OK".
                res=self.errors[resdll]
            else: #For all other errors, return "error code + error description"
                res="error code "+str(resdll)+", "+self.errors[resdll]
        else:
            res="unknown error code ("+str(resdll)+")"

        return res

    def load_spec_dll(self):
        """
        res=load_dll()
        Load the spectrometer control dll, and store the handler into dll_handler

        returns
        <res>: string with the result of the operation. It can be "OK" or an error description.
        """
        res="OK"
        self.logger.info("Loading dll: "+self.dll_path)

        #Check if file exists:
        if not os.path.exists(self.dll_path):
            res="The dll file does not exist: "+self.dll_path
            self.logger.error(res)
            return res

        try:
            if self.dll_handler is None:
                self.dll_handler = windll.LoadLibrary(self.dll_path)
            #Note: Python knows if a dll has been loaded before or not: When loading an already loaded dll,
            #it returns the same memory address of the already loaded dll.
            if self.debug_mode>0:
                self.logger.debug("dll_handler: "+str(self.dll_handler))
        except Exception as e:
            res="Exception happened while loading the dll: "+str(e)
            self.logger.exception(e)

        return res

    def initialize_dll(self):
        """
        res=self.initialize_dll()

        Initializes the dll with AVS_Init().
        This function tries to open a communication port for all connected devices.

        returns:
        <res>: string with the result of the operation. It can be "OK" or an error description.
        """
        res="OK"

        self.logger.info("Initializing spec "+self.alias+" dll...")

        if self.simulation_mode:
            pass
        else:
            try:
                #Call AVS_Init()
                resdll=self.dll_handler.AVS_Init(0) #0=USB connected devices, 256=ethernet connected devices
                #resdll will return the number of Avantes USB spectrometer devices found in the system (>=0), or an error code (<0)
                if resdll>=0: #Correct answer
                    if resdll>0: #Found more than 0 devices connected -> OK
                        pass
                    else: #Found 0 devices connected -> NOK
                        res="Could not connect spectrometer "+self.alias+" (sn="+self.sn+"), "
                        res+="\n"+"found 0 connected devices of this type (Avantes)."
                        self.logger.error("initialize_dll, "+res.replace("\n"," "))
                else: #Error answer (resdll<0)
                    res=self.get_error(resdll)
                    res="Could not initialize dll for spec "+self.alias+": "+res
                    self.logger.error("initialize_dll, "+res)
            except Exception as e:
                res="Exception happened while initializing the dll: "+str(e)
                self.logger.exception(e)

        self.error=res #Update last error.

        return res

    def enable_dll_logging(self,enable):
        """
        Enables or disables writing dll debug information to a log file,
        called "avaspec.dll.log", located in your user directory.
        Implemented for Windows only.

        params:
         <enable>: Boolean, True enables logging, False disables logging
        return:
         <res>: "OK" or an error description
        """
        res="OK"
        if self.simulation_mode:
            pass
        else:
            resdll=self.dll_handler.AVS_EnableLogging(c_bool(enable))
            if resdll==1: #Correct answer for this function
                res="OK"
                if enable:
                    self.logger.info("dll logging enabled")
                else:
                    self.logger.info("dll logging disabled")
            else: #Incorrect answer
                if enable:
                    res="Could not enable logging. Wrong dll answer ("+str(resdll)+")"
                else:
                    res="Could not disable logging. Wrong dll answer ("+str(resdll)+")"
                self.logger.error(res)

        self.error=res
        return res

    def get_number_of_devices(self):
        """
        res,ndev=self.get_number_of_devices()

        Returns the number of spectrometers connected to the computer.

        :return:
        <res>: string with the result of the operation. It can be "OK" or an error description.
        <ndev>: number of spectrometers connected to the computer (integer)
        """
        res="OK"
        ndev=0

        if self.simulation_mode:
            self.logger.debug("get_number_of_devices, simulating a connected spectrometer device.")
        else:
            try:
                ndev=self.dll_handler.AVS_UpdateUSBDevices()
                if ndev==0:
                    res="Cannot detect any Avantes spectrometer connected through USB."
                    self.logger.error("get_number_of_devices, "+res)
                else:
                    self.logger.info("get_number_of_devices, found "+str(ndev)+" Avantes spectrometers connected through USB.")
            except Exception as e:
                res="Exception happened while getting the number of spectrometers: "+str(e)
                self.logger.exception(e)

        self.error=res #Update last error.

        return res,ndev

    def get_all_devices_info(self,ndev):
        """
        res,l_pData_all=self.get_all_devices_info(ndev)

        Returns the device information of all connected spectrometers.

        params:
            <ndev>: Number of connected devices (integer)

        return:
            <res>: string with the result of the operation. It can be "OK" or an error description.
            <l_pData_all>: device information of all connected spectrometers (structure)
        """

        res="OK"
        self.logger.info("Getting device information of all connected spectrometers...")

        #create an array of AVSIdentities
        class AvsIDArray(Structure):
            _fields_=[]
            for i in range(ndev):
                _fields_.append(("a"+str(i),AvsIdentityType))

        #get device information of all connected specs
        l_pData_all=AvsIDArray() #= Devices information array

        if self.simulation_mode:
            pass
        else:
            l_RequiredSize=c_uint(sizeof(AvsIdentityType)*ndev)
            l_Size=l_RequiredSize
            resdll=self.dll_handler.AVS_GetList(l_Size,byref(l_RequiredSize),byref(l_pData_all))
            #Will return the number of devices (elements) in the device information array (>=0), or an error code (<0).
            if resdll>0:
                #Print the serial numbers of the connected spectrometers:
                sns=[]
                for i in range(ndev):
                    i_l_pData=l_pData_all.__getattribute__("a"+str(i))
                    sn=i_l_pData.SerialNumber
                    if sys.version_info[0] > 2: #in python 3.x sn comes as bytestring
                        sn=sn.decode("utf8")
                    sns.append(sn)
                self.logger.info("Serial Numbers of spectrometers found: "+", ".join(sns))
            elif resdll==0:
                res="Could not get device information of any of the USB connected spectrometers. (Connection lost?)."
                self.logger.error("get_all_devices_info, "+res)
            else: #Error answer (resdll<0)
                res=self.get_error(resdll)
                res="Could not get device information of the USB connected spectrometers: "+res
                self.logger.error("get_all_devices_info, "+res)

        self.error=res #Update last error.

        return res,l_pData_all

    def find_spec_info(self,ndev,l_pData_all):
        """
        res=self.find_spec_info(ndev,l_pData_all)

        Get spec parameters of the spectrometer with serial number self.sn.

        params:
         <ndev>: Number of connected devices (integer), as it comes from get_number_of_devices()
         <l_pData_all>: device information of all connected spectrometers (structure)

        return:
        <res>: string with the result of the operation. It can be "OK" or an error description.
        <l_pData>: spectrometer data (structure)
        """
        res="OK"

        self.logger.info("find_spec_info, getting parameters of spectrometer "+str(self.alias)+" with serial number "+str(self.sn)+"...")
        if self.simulation_mode:
            l_pData=None
        else:
            l_pData=None
            specs_sn_found=[] #Append the serial numbers here
            for i in range(ndev):
                i_l_pData=l_pData_all.__getattribute__("a"+str(i))
                #Get the serial number of the i-th spectrometer:
                i_spec_sn=i_l_pData.SerialNumber
                if sys.version_info[0] < 3: #in python 2.x i_spec_sn comes as str
                    i_spec_sn_str=i_spec_sn
                else:#in python 3.x i_spec_sn comes as bytestring
                    i_spec_sn_str=i_spec_sn.decode("utf8")
                specs_sn_found.append(i_spec_sn_str)
                if i_spec_sn_str==self.sn:
                    l_pData=i_l_pData
                    break
            else: #no break happened
                res="Could not find parameters of spectrometer with serial number "+self.sn+"."
                if len(specs_sn_found)>0:
                    res+="\n"+" Only found spectrometers with serial numbers: "+str(specs_sn_found)
                self.logger.error("find_spec_info, "+res)

            self.error=res

        return res,l_pData

    def activate_spec(self,l_pData):
        """
        res=self.activate_spec(l_pData)

        Activates an Avanntesspectrometer.

        params:
            <l_pData>: spectrometer data (structure), as it comes from find_spec_info()

        return:
            <res>: string with the result of the operation. It can be "OK" or an error description.
            <spec_id>: spectrometer id (byte string). It is a device identificator to be used in subsequent DLL calls.
        """
        res="OK"
        spec_id=None
        self.logger.info("Activating spectrometer "+self.alias+"...")

        if self.simulation_mode:
            pass
        else:
            try:
                resdll=self.dll_handler.AVS_Activate(byref(l_pData))
                if resdll==1000:
                    res="Could not activate spectrometer "+self.alias+". Error: "+self.get_error(resdll)
                    self.logger.error("activate_spec, "+res)
                else:
                    spec_id=resdll
                    self.logger.info("Spectrometer "+self.alias+" activated. (spec_id="+str(spec_id)+")")
            except Exception as e:
                res="Exception happened while activating spectrometer "+self.alias+": "+str(e)
                self.logger.exception(e)

        self.error=res

        return res,spec_id

    def get_device_type(self):
        """
        res,devtype=self.get_device_type()

        Returns the device type of the spectrometer.
        returns:
            <res>: string with the result of the operation. It can be "OK" or an error description.
            <devtype>: string with the device type of the spectrometer.
        """


        devtype=c_byte()
        resdll=self.dll_handler.AVS_GetDeviceType(self.spec_id,byref(devtype))
        res=self.get_error(resdll)
        if res!="OK":
            devtype=devtypes[0] #= TYPE_UNKNOWN
            res="Could not get device type of spec "+self.alias+". Error: "+res
            self.logger.error("get_device_type, "+res)
        else:
            devtype=devtype.value
            if devtype in devtypes:
                devtype=devtypes[devtype]
            else:
                devtype=devtypes[0] #= TYPE_UNKNOWN
            self.logger.info("Device type of spec "+self.alias+": "+str(devtype))

        self.error=res
        return res,devtype

    def get_detector_name(self):
        """
        res,sname=self.AVS_GetDetectorName()
        Returns the name of the detector inside the spectrometer.

        return:
            <res>: string with the result of the operation. It can be "OK" or an error description.
            <sname>: string with the name of the detector inside the spectrometer.
        """
        sname=(c_char * 64)()
        resdll=self.dll_handler.AVS_GetDetectorName(self.spec_id,c_int(0),byref(sname))
        res=self.get_error(resdll)
        if res!="OK":
            sname=""
            res="Could not get detector name of spec "+self.alias+". Error: "+res
            self.logger.error("AVS_GetDetectorName, "+res)
        else:
            sname=sname.value
            if sys.version_info[0] > 2: #in python 3.x sname comes as bytestring
                sname=sname.decode("utf8") #convert to string

            self.logger.info("Detector inside spectrometer: "+sname)

        self.error=res
        return res,sname

    def get_version_info(self):
        """
        Returns FPGA version, Firmware version, and Library version of the used system.
        Note that the library does not check the size of the buffers allocated by the caller!

        return:
            <res>: string with the result of the operation. It can be "OK" or an error description.
            <FPGAversion>: FPGA version of the used system.
            <FWversion>: Firmware version of the used system.
            <DLLversion>: Library version of the used system.
        """
        FPGA_version=(c_char * 16)()
        FW_version=(c_char * 16)()
        DLL_version=(c_char * 64)()
        resdll=self.dll_handler.AVS_GetVersionInfo(self.spec_id,byref(FPGA_version),byref(FW_version),byref(DLL_version))
        res=self.get_error(resdll)
        if res!="OK":
            FPGA_version=FW_version=DLL_version=""
            res="Could not get version info of spec "+self.alias+". Error: "+res
            self.logger.error("get_version_info, "+res)
        else:
            if sys.version_info[0] > 2: #in python 3.x, these outputs comes as bytestring
                FPGA_version=FPGA_version.value.decode("utf8")
                FW_version=FW_version.value.decode("utf8")
                DLL_version=DLL_version.value.decode("utf8") #convert to string
            else:
                FPGA_version=str(FPGA_version.value)
                FW_version=str(FW_version.value)
                DLL_version=str(DLL_version.value)
            self.logger.info("FPGA version: "+FPGA_version)
            self.logger.info("Firmware version: "+FW_version)
            self.logger.info("Dll library version: "+DLL_version)

        self.error=res
        return res,FPGA_version,FW_version,DLL_version

    def get_device_config(self):
        """
        res=self.get_device_config()
        Get the device config parameters of the spectrometer.
        (Testing function, just to confirm if some parameters are being set correctly)
        """
        devconfig=DeviceConfigType()
        size=c_uint32(63484)
        reqsize=c_uint32()

        #Get device parameters:
        resdll=self.dll_handler.AVS_GetParameter(self.spec_id,size,byref(reqsize),byref(devconfig))
        if resdll!=0: #0 at success, -1 at error. (This function does not return any detailed error code)
            res="Could not get device config parameters of spec "+self.alias+". Error code: "+str(resdll)
            res+=" (spec_id="+str(self.spec_id)+", reqsize="+str(reqsize.value)+", size="+str(size.value)+")"
            self.logger.error("get_device_config, "+res)
        else:
            res="OK"
            self.logger.info("get_device_config, got device config parameters of spec "+self.alias+".")

        self.error=res
        return res

    def set_device_config(self,devconfig):
        """
        res=self.set_device_config(devconfig)
        Set the device config parameters of the spectrometer.
        (Use carefully)
        params:
            <devconfig>: DeviceConfigType structure with the new parameters to be set. (As given by get_device_config())
        """

        #Get device parameters:
        resdll=self.dll_handler.AVS_SetParameter(self.spec_id,devconfig)
        res=self.get_error(resdll)
        if res!="OK":
            res="Could not set device config parameters of spec "+self.alias+". Error: "+res
            self.logger.error("set_device_config, "+res)
        else:
            self.logger.debug("set_device_config, set device config parameters of spec "+self.alias+".")

        self.error=res
        return res

    def reset_device_config(self):
        """
        res=self.reset_device_config()
        Resets onboard device parameter section to its factory defaults. This command will result in
        the loss of all user-specific device configuration settings which is set through the AvaSpec
        function AVS_SetParameter()
        Note: this will erase the OEM section of the EEPROM.
        """
        resdll=self.dll_handler.AVS_ResetParameter(self.spec_id)
        res=self.get_error(resdll)
        if res!="OK":
            res="Could not set default device config parameters of spec "+self.alias+". Error: "+res
            self.logger.error("reset_device_config, "+res)
        else:
            self.logger.debug("reset_device_config, set default device config parameters of spec "+self.alias+".")

        self.error=res
        return res

    def get_number_of_pixels(self):
        """
        res=self.get_number_of_pixels()
        Get the number of pixels of the device specified by self.spec_id.
        """
        res="OK"
        npix=c_ushort()

        #Check device id is not none:
        if self.spec_id==None:
            res="Could not get number of pixels of spec "+self.alias+", device is not initialized."
            self.logger.error("get_number_of_pixels, "+res)
        else:
            try:
                resdll=self.dll_handler.AVS_GetNumPixels(self.spec_id,byref(npix))
                res=self.get_error(resdll)
                if res=="OK":
                    pass
                else:
                    res="Could not get number of pixels of spec "+self.alias+", error: "+res
                    self.logger.error("get_number_of_pixels, "+res)
            except Exception as e:
                res="Exception happened while getting the number of pixels of spec "+self.alias+": "+str(e)
                self.logger.exception(e)

        self.error=res
        return res,npix.value

    def set_sensitivity(self,mode=-1):
        """
        res=self.set_sensitivity_Ava1(mode)
        Selects between LowNoise and HighSensitivity mode for certain detectors (InGas).
        <mode> unsigned integer, 0 sets LowNoise mode, 1 sets HighSensitivity mode
        return: res
        """
        modes={0:"low noise",1:"high sensitivity"}

        if mode in modes:

            self.logger.info("Setting sensitivity mode to: "+str(modes[mode]))

            try:
                resdll=self.dll_handler.AVS_SetSensitivityMode(self.spec_id,mode)
                res=self.get_error(resdll)
                if res!="OK":
                    res="Could not set sensitivity mode. Error: "+res
                    self.logger.error(res)
            except Exception as e:
                res="Exception happened while setting sensitivity mode: "+str(e)
                self.logger.error(res)
                self.logger.exception(e)
        else:
            res="Could not set sensitivity mode, mode="+str(mode)+" not allowed. (Valid modes are "+str(modes.keys())+")"
            self.logger.error(res)

        self.error=res
        return res

    def register_status_callback(self):
        """
        res=self.register_status_callback()
        Register a callback function to be called when the status of the spectrometer changes.
        """
        res="OK"
        resdll=self.dll_handler.AVS_Register(self.conn_status_cb_function)
        #it returns True or False.
        if not resdll:
            res="Could not register status callback of spec "+self.alias
            self.logger.error("register_status_callback, "+res)
        else:
            self.logger.info("register_status_callback, registered status callback of spec "+self.alias+".")
        self.error=res
        return res


    def deactivate(self,ignore_errors=False):
        """
        Closes the dll communication of a previously activated spectrometer.
        It will remove the self.spec_id handler.
        """
        res="OK"

        if self.dll_handler is None:
            res="Could not deactivate spectrometer "+self.alias+". Dll handler is not initialized."
            self.logger.error("deactivate, "+res)
            return res

        if self.spec_id is None:
            res="Could not deactivate spectrometer "+self.alias+". Device identifier is not initialized."
            self.logger.error("deactivate, "+res)
            return res

        self.logger.info("Deactivating spectrometer "+self.alias+"...")
        if self.simulation_mode:
            pass
        else:
            try:
                resdll=self.dll_handler.AVS_Deactivate(self.spec_id) #This function returns 1 true or 0 false (=device identifier not found).
                if resdll!=1:
                    if resdll==0:
                        res="Could not deactivate spectrometer "+self.alias+". Device identifier not found."
                    else:
                        res="Could not deactivate spectrometer "+self.alias+". Error: "+self.get_error(resdll)
                    if ignore_errors:
                        self.logger.warning("deactivate, "+res)
                        res="OK"
                    else:
                        self.logger.error("deactivate, "+res)
                else:
                    self.logger.info("Spectrometer "+self.alias+" deactivated.")

            except Exception as e:
                res="Exception happened while deactivating spectrometer "+self.alias+": "+str(e)
                if ignore_errors:
                    self.logger.warning("deactivate, "+res)
                    res="OK"
                else:
                    self.logger.exception(e)

        self.spec_id=None #Release spec_id handler

        return res

    def close_spec_dll(self,ignore_errors=False,force_dofree=False):
        """
        Finalize the avantes dll communication and release its internal storage. (clears dll_handler)
        Note: call to this function only once at exit time, once all spectrometers have been disconnected!.
        Otherwise, all connected spectrometers will be unmanageable after calling this function.

        params:
            <ignore_errors>: (boolean) Set this to True to ignore any error that happens during the closing process.
            <force_dofree>: (boolean) By default the function won't allow to close the dll session if there is any
            spectrometer still connected. This parameter allows to skip this safety check.
        """
        res="OK"

        if self.dll_handler is None:
            self.logger.info("Avantes dll communication is already closed.")
            return res
        if len(Avantes_Spectrometer_Instances)!=0 and not force_dofree:
            #Instances will be deleted from Avantes_Spectrometer_Instances at disconnection time.
            #So it is needed first to disconnect all spectrometers, and then to close the dll communication.
            self.logger.info("Not closing the dll communication because there is another avantes spectrometer that is still needing it.")
        else: #This part will be executed only when all specs have been already disconnected.
            try:
                self.logger.info("Closing avantes dll communication with AVS_Done()")
                resdll=self.dll_handler.AVS_Done()
                if resdll==0:
                    self.logger.info("Avantes dll communication closed.")
                else:
                    res="Could not close dll. resdll="+str(resdll)
                    if ignore_errors:
                        self.logger.warning("close_spec_dll, "+res)
                        res="OK"
                    else:
                        self.logger.error("close_spec_dll, "+res)
            except Exception as e:
                res="Exception happened while closing the avantes dll: "+str(e)
                if ignore_errors:
                    self.logger.warning("close_spec_dll, "+res)
                    res="OK"
                else:
                    self.logger.exception(e)
            self.dll_handler=None #Release dll_handler
        return res


    #Auxiliary functions related to measurements
    def reset_spec_data(self):
        """
        This function is called at the start of every new measurement.
        """

        #Ensure read_data_queue and handle_data_queue are empty before start using them.
        while not self.read_data_queue.empty():
            _=self.read_data_queue.get()

        while not self.handle_data_queue.empty():
            _=self.handle_data_queue.get()


        self.ncy_read=0 #Current number of cycles measured and read from the spectrometer roe
        self.ncy_handled=0 #Current number of cycles handled
        self.ncy_saturated=0 #Number of saturated measurements. (only active pixels checked)
        self.sy=np.zeros(self.npix_active,dtype=np.float64) #Sum of the counts
        self.syy=np.zeros(self.npix_active,dtype=np.float64) #Sum of the squared counts
        self.sxy=np.zeros(self.npix_active,dtype=np.float64) #Sum of the meas index by the counts
        self.sy_blind_left=np.zeros(self.npix_blind_left,dtype=np.float64) #Sum of the counts of the blind pixels at left side of the detector
        self.syy_blind_left=np.zeros(self.npix_blind_left,dtype=np.float64) #Sum of the squared counts of the blind pixels at left side of the detector
        self.sxy_blind_left=np.zeros(self.npix_blind_left,dtype=np.float64) #Sum of the meas index by the counts of the blind pixels at left side of the detector
        self.arrival_times=[] #List of arrival times of the measurements (Time in which the callback function was called)
        self.meas_start_time=0 #Unix time in seconds when the measurement started
        self.meas_end_time=0 #Unix time in seconds when the measurement ended (data arrival time of the last measured cycle)
        self.data_handling_end_time=0 #Unix time in seconds when the data handling ended (all cycles received + handled + final data handling)



    def set_store_to_ram_ncy(self,ncy):
        """
        Set the number of cycles to store to ram for every call to AVS_MeasureCallback.
        (Only must be used when store_to_ram=True)
        """
        #Update parameters list:
        if self.store_to_ram:
            self.parlist.m_Control_m_StoreToRam=c_uint16(int(ncy))
        else:
            self.parlist.m_Control_m_StoreToRam=c_uint16(0)
        #Call AVS_PrepareMeasure to update the new config parameters:
        resdll=self.dll_handler.AVS_PrepareMeasure(self.spec_id,byref(self.parlist))
        res=self.get_error(resdll)
        if res!="OK":
            res="Could not set ncy to "+str(ncy)+". Error: "+res
            self.logger.error(res)

        self.error=res
        return res

    #Auxiliary functions for data retrieval (read data from spectrometer)

    def data_arrival_watchdog(self):
        """
        This function will run permanently in a side thread, and is constantly waiting to have something in the self.read_data_queue.
        (This queue is filled by the dll callback function: whenever a measurement is ready to be read, something is written
         into this queue).
        As soon as it gets something, it will proceed to get the data from the spectrometer,
        and put this data into the self.handle_data_queue.

        When disconnecting the spectrometer, the disconnect function will send a "None" to the self.read_data_queue,
        to signal that the data arrival watchdog thread must be finished.
        """

        #Ensure read_data_queue is empty before start using it.
        while not self.read_data_queue.empty():
            _=self.read_data_queue.get()

        #Start infinite data arrival monitoring loop:
        while True:
            lerror,arrival_time=self.read_data_queue.get()

            if lerror is None: #Exit flag -> data arrival watchdog thread must be finished.
                self.logger.info("Exiting data arrival watchdog thread of spectrometer "+self.alias+"...")
                break

            elif lerror=="abort":
                # Order to abort due to saturation -> indicate to the spectrometer that measurement
                # has been aborted, so that no more signals are received to read data.
                if self.debug_mode>=2:
                    self.logger.debug("data_arrival_watchdog, sending order to abort measurement")
                _=self.abort()

                #Ignore all data arrival signals from now on. Empty the read_data_queue.
                while not self.read_data_queue.empty():
                    _=self.read_data_queue.get()

            else: #Normal data arrival -> check error and get data:

                if lerror==0: #No error reported by dll
                    #self.logger.debug("data_arrival_watchdog, reading data of spec "+self.alias+"...")
                    res,rc,rc_blind_left=self.read_data()

                    if res=="OK" and self.docatch:
                        self.ncy_read+=1 #Increment number of measurements read
                        if self.debug_mode>=3:
                            self.logger.debug("data_arrival_watchdog, got spec "+self.alias+" data, nmeas read="+str(self.ncy_read)+"/"+str(self.ncy_requested)+".")
                        #Add received data to the data handling queue. (Will be handled by the data handling watchdog thread):
                        #A tuple is used to pass the measurement index and the data to the queue: (meas_done,raw_counts).
                        #raw_counts is another tuple that contains the raw counts of the active pixels and the raw counts of the blind pixels (if any).
                        self.handle_data_queue.put((deepcopy(self.ncy_read), #Index of the cycle read
                                                    arrival_time, #arrival time of the cycle
                                                    (rc,rc_blind_left))) #raw data of the cycle read

                        #Start next emulated measurement timer, in case of simulation mode and not store_to_ram mode:
                        if self.simulation_mode and not self.store_to_ram and self.ncy_read<self.ncy_requested:
                            self.simulated_measurement_timer=threading.Timer(self.it_ms*1e-3,measure_callback,args=((self.spec_id,),(0,)))
                            self.simulated_measurement_timer.start()

                else: #error reported by dll
                    res=self.get_error(lerror)
                    res="Error in data arrival of spec "+self.alias+": "+res
                    self.logger.error("data_arrival_watchdog, "+res)

                #If error,
                if res!="OK":
                    #update last error:
                    self.error=res
                    #signal that the measurement is "complete" to unblock the main thread in case of a measure_blocking call:
                    self.internal_meas_done_event.set()
                    #If an external_meas_done_event is set, signal it too:
                    if self.external_meas_done_event is not None and not self.recovering:
                        self.external_meas_done_event.set()


    def read_data(self):
        """
        This function will be called by the data arrival watchdog, when a measurement is ready to be read.
        It will be called for every cycle measured, so it must be fast.
        returns:
            <res>: string with the result of the operation. It can be "OK" or an error description.
            <rc>: raw counts of the active pixels,
            <rc_blind_left> raw counts of the blind pixels on the left side of the detector (if any)
        """
        self.busy=True
        if self.simulation_mode:
            #Create ramdom data:
            rc=np.random.rand(self.npix_active)*1000
            rc_blind_left=np.random.rand(self.npix_blind_left)*1000
            res="OK"
        else:
            #Create input buffers:
            a_pTimeLabel=c_uint() #ticks count last pixel of spectrum is received by microcontroller ticks in 10 uS units since spectrometer started
            rc=(c_double*self.npix_active)() #input buffer where to store raw counts
            rc_blind_left=(c_double*self.npix_blind_left)() #input buffer where to store the raw counts of the left-side blind pixels.

            #Get active pixels data
            resdll=self.dll_handler.AVS_GetScopeData(self.spec_id,byref(a_pTimeLabel),byref(rc))

            res=self.get_error(resdll)
            if res!="OK":
                res="Could not get data from spec "+self.alias+". Error: "+res
                self.logger.error("read_data, "+res)
            else:
                #get blind pixels data:
                if self.npix_blind_left>0: #Only if there are blind pixels
                    resdll=self.dll_handler.AVS_GetDarkPixelData(self.spec_id,byref(rc_blind_left))
                    #Note: This function extract the blind pixels signal from the last measured spectrum.
                    #It returns True if ok, or False otherwise.
                    if not resdll:
                        res="Could not get left-side blind pixels data from spec "+self.alias+". Error: "+res
                        self.logger.error("read_data, "+res)
        self.busy=False
        return res,rc,rc_blind_left


    #Auxiliary functions for data handling

    def data_handling_watchdog(self):
        """
        This function will run permanently in a side thread, and is constantly waiting to have something in the self.handle_data_queue.
        (This queue is filled by the data arrival watchdog thread: whenever a measurement has been read and is ready to be handled,
         something is written in this queue).
        As soon as it gets something, it will proceed to handle the data, and check if the measurement is complete.
        If the measurement is complete, it will signal the self.internal_meas_done_event, which will unblock the main thread, in
        case of a measure_blocking call.
        """

        #Ensure data_handling_queue is empty before start using it.
        while not self.handle_data_queue.empty():
            _=self.handle_data_queue.get()

        #Start the infinite data handling loop:
        while True:
            (ncy_read, arrival_time, (rc,rc_blind_left))=self.handle_data_queue.get()
            if ncy_read is None: #Exit flag -> data arrival watchdog thread must be finished.
                self.logger.info("Exiting data handling watchdog thread of spectrometer "+self.alias+"...")
                break
            elif not self.docatch:
                #Do not handle more data from now on.
                while not self.handle_data_queue.empty():
                    _=self.handle_data_queue.get()
            else: #Normal data arrival -> handle cycle data:
                #Add arrival time to the "effective" arrival times list
                # which contains only the arrival times of the really handled cycles.
                self.arrival_times.append(arrival_time)
                issat=self.handle_cycle_data(ncy_read,rc,rc_blind_left)
                if issat and self.abort_on_saturation:
                    self.logger.info("data_handling_watchdog, saturation detected in spec "+self.alias+
                                ", for nmeas read ="+str(ncy_read)+"/"+str(self.ncy_requested)+
                                ". Aborting due to saturation...")

                    self.docatch=False #Ignore the data arrival events from now on.
                    #This will:
                    # - Prevent the measure callback function to put anything into the read_data_queue. So that, new
                    #   data ready signals coming from the spectrometer dll will be ignored.
                    # - Prevent the read_data_watchdog to put more (read) data into the handle_data_queue.
                    # - Prevent the handle_data_watchdog to handle more data.

                    #Put a special signal into the read_data_queue, so that the read_data_watchdog sends
                    #an abort order to the spectrometer when possible. In this way the spectrometer is aware that
                    #it has to stop measuring / sending data ready signals to the pc.
                    #This is done through the read_data_watchdog in order to avoid concurrency:
                    # if an abort order is sent while data is being read, the dll might return an error.
                    self.read_data_queue.put(("abort",0)) #lerror,arrival_time

                    #Do not handle more data. (Discard any pending data handling orders)
                    while not self.handle_data_queue.empty():
                        _=self.handle_data_queue.get()

                    #Finish the measurement:
                    self.measurement_done()
                else:
                    if self.debug_mode>=3:
                        self.logger.debug("data_handling_watchdog, handled spec "+self.alias+" data, ncy handled="+str(self.ncy_handled)+"/"+str(self.ncy_requested)+".")
                    #If measurement is complete:
                    if self.ncy_handled==self.ncy_requested:
                        self.measurement_done()

    def handle_cycle_data(self,ncy_read,rc,rc_blind_left):
        """
        Handle the measurement cycle data
        This function will be called by the data handling watchdog, when a measurement has been already read from the
        spectrometer, and is ready to be handled.
        Data handling means convert the read raw counts from whatever format it comes (ctypes array) to a numpy array,
        and applying any needed post processing, to have the data in raw counts units.
        Then it is checked if the data is saturated, and if so, the saturated_meas_counter is incremented.
        Finally, the data is accumulated in the sy, syy, and sxy variables, that are used to calculate the mean and standard deviation
        at the end of the measurement.

        params:
            <ncy_read>: read measurement number (from 1 to requested_nmeas)
            <rc>: raw counts of the active pixels (ctypes array)
            <rc_blind_left>: raw counts of the blind pixels on the left side of the detector (if any) (ctypes array)

        returns:
            <issat>: boolean, True if the last handled data is saturated, False otherwise.
        """

        #Convert raw counts ctypes array to numpy array
        rc=np.array(rc)
        #Convert to float64
        rc=rc.astype(np.float64)
        #Apply discriminator factor:
        rc=rc*float(self.discriminator_factor)
        rcmax=rc.max()
        rcmin=rc.min()
        #TODO: Check consistency of the data. (eg. all elements >0, no nans, etc.)
        if rcmin<0:
            self.logger.warning("handle_cycle_data, negative counts detected in spec "+self.alias+" data.")

        #Detect saturation:
        issat=rcmax>=self.eff_saturation_limit
        if issat and self.abort_on_saturation:
            #Do not add cycle data to accumulated data in this case.
            #No more cycles will be handled from now on.
            #This cycle data won't be used.
            #(in the accumulated data there won't be any saturated cycle)
            return issat #-> Quit

        else: #Continue even if saturation is detected:

            #Add cycle data to accumulated data for active pixels:
            self.sy=self.sy+rc
            self.syy=self.syy+rc**2
            self.sxy=self.sxy+(ncy_read-1)*rc

            #Do the same for blind pixels (if any):
            if len(rc_blind_left)>0:
                #Convert ctypes array to numpy array
                rc_blind_left=np.array(rc_blind_left)
                #Convert to float64
                rc_blind_left=rc_blind_left.astype(np.float64)
                #Apply discriminator factor:
                rc_blind_left=rc_blind_left*float(self.discriminator_factor)
                #Add data to accumulated data:
                self.sy_blind_left=self.sy_blind_left+rc_blind_left
                self.syy_blind_left=self.syy_blind_left+rc_blind_left**2
                self.sxy_blind_left=self.sxy_blind_left+(ncy_read-1)*rc_blind_left

            self.ncy_handled+=1
            if issat:
                self.ncy_saturated+=1

        return issat

    def measurement_done(self):
        """
        Final actions to be done when a measurement is complete.
        """
        self.meas_end_time=self.arrival_times[-1] #Time in which the spectrometer indicated to the pc that
        # the last cycle was finished, and it was ready to be read.
        #Calculate mean, standard deviation and rms to a fitted straight line (for active pixels):
        x=np.arange(self.ncy_handled)
        _,self.rcm,self.rcs,self.rcl=calc_msl(self.alias,x,self.sxy,self.sy,self.syy)
        if self.npix_blind_left>0: #same for blind pixels
            _,self.rcm_blind_left,self.rcs_blind_left,self.rcl_blind_left=calc_msl(self.alias,x,self.sxy_blind_left,self.sy_blind_left,self.syy_blind_left)
            self.rcl_blind_left=self.rcs_blind_left #replace rcl by rcs for blind pixels

        if self.debug_mode>=1:
            self.logger.debug("Measurement done for spec "+self.alias)
        self.data_handling_end_time=spec_clock.now()
        self.measuring=False
        self.docatch=False
        #Signal that the measurement is complete:
        self.internal_meas_done_event.set()
        if self.external_meas_done_event is not None:
            if not self.recovering:
                self.external_meas_done_event.set()


    #Auxiliary functions related to performance & statistics

    def performance_test(self,fpath=""):
        """
        This function performs a certain number of sequential measurements at different integration times,
        and different number of cycles, and calculates the performance statistics for each of them.
        The results are stored into a txt file, where the first line is the header, and the following columns contains:
        -the first column contains the integration times,
        -the second column contains the number of cycles,
        -the third column contains the real measurements duration (discarding the cycle handling time),
        -the fourth column contains the "mean" calculated cycle delay time (cdt_mean) for the respective integration time and number of cycles,
          calculated as cdt_mean=(Real_dur_measurements-ncy*IT)/ncy
        -the fifth column contains the "median" cycle delay time (cdt_median), calculated as
          cdt_median=((median delta of data arrival events)-IT). This is only possible when we have a data arrival time for
          every measured cycle. If not, the cdt_median will be equal to cdt_mean.  for each integration time and number of cycles, calculated as cdt_median=((median delta of data arrival events)-IT)

        params:
            <fpath>: (string) Path to the folder where the results file will be stored.
        returns:
            <res>: (string) "OK" if the performance test was successful, or an error description if not.
            <presults>: (np array) 2D array with the performance results.

        """
        res="OK"
        presults=np.array([])
        self.logger.info("Doing performance test - analyzing cycle delay time behavior with respect ncy and IT")
        its=[]
        ncys=[]
        real_dur_meass=[]
        cdts_mean=[]
        cdts_median=[]

        test_start_time=spec_clock.now()

        for it in self.performance_test_it_ms_list:
            it=round(it,3) #avoid numerical errors
            self.set_it(it_ms=it)
            sleep(it*1e-3) #ms->s (Wait to have new IT applied)
            for ncy in self.performance_test_ncy_list:
                ncy=int(ncy)
                res=self.measure_blocking(ncy=ncy)
                if res!="OK":
                    break
                cdt_mean,cdt_median,real_dur_meas,_,_,_=self.calc_performance_stats(showinfo=False) #get cycle delay time
                its.append(it)
                ncys.append(ncy)
                real_dur_meass.append(real_dur_meas)
                cdts_mean.append(cdt_mean)
                cdts_median.append(cdt_median)
                if self.store_to_ram:
                    self.logger.info("Spec "+self.alias+", IT="+str(it)+" ms, ncy="+str(ncy)+", cdt_mean="+str(cdt_mean)+" ms/cy")
                else:
                    self.logger.info("Spec "+self.alias+", IT="+str(it)+" ms, ncy="+str(ncy)+", cdt_mean="+str(cdt_mean)+" ms/cy, cdt_median="+str(cdt_median)+" ms/cy")

        if res=="OK":
            test_end_time=spec_clock.now()
            test_duration=test_end_time-test_start_time #s
            self.logger.info("Performance test duration: "+str(test_duration)+" s")

            #Put the performance results all together into a 2D np array, and write it into a file:
            if self.store_to_ram:
                presults=np.array([its,ncys,real_dur_meass,cdts_mean]).T
                header="IT[ms]; ncy; real_dur [ms]; cdt_mean[ms/cy]"
                ptest_filename=self.sn+"_performance_test_STR.txt"
            else:
                presults=np.array([its,ncys,real_dur_meass,cdts_mean,cdts_median]).T
                header="IT[ms]; ncy; real_dur [ms]; cdt_mean[ms/cy]; cdt_median[ms/cy]"
                ptest_filename=self.sn+"_performance_test.txt"
            ptest_filepath=os.path.join(fpath,ptest_filename)
            with open(ptest_filepath,"w") as f:
                np.savetxt(f,presults,fmt="%.4f",delimiter=";",header=header,comments="#")

        if res=="OK":
            self.logger.info("Finished performance test")
        else:
            res="Error during performance test: "+res
            self.logger.error(res)

        self.error=res
        return res,presults

    def calc_performance_stats(self,showinfo=True):
        """
        Calculate the performance statistics of the last measurement.
        -Measurements duration
        -Final data handling duration
        -Total duration
        -Determination of the mean and median cycle delay time
        -In case the cycles were measured one by one, the maximum and minimum deltas between the data arrival times.
        """
        real_dur_meas=1000.0*(self.meas_end_time-self.meas_start_time) #real duration of measurements [ms]
        real_dur_fdh=1000.0*(self.data_handling_end_time-self.meas_end_time) #real duration of the final data handling, [ms]
        real_dur_total=real_dur_meas+real_dur_fdh # real total duration (measurements + final data handling) [ms]
        expected_min_dur_meas=self.ncy_requested*self.it_ms #expected minimum duration of the measurements, [ms]. (=ncy*IT)
        #expected_min_dur_fdh=0 #expected minimum duration of final data handling, [ms]
        #expected_min_dur_tot=expected_min_dur_meas+expected_min_dur_fdh #expected minimum total duration, [ms]
        cdt_mean=max(0,(real_dur_meas-expected_min_dur_meas)/self.ncy_requested) #Mean cycle delay time of last measurement [ms/cy]
        deltas_min=np.nan
        deltas_max=np.nan

        #Calc the median cycle delay time (will be used to estimate the duration of a future measurement, as dur=ncy*(IT+cdt_median))
        #2 possible cases (for avantes):
        #1) Measured ncy cycles one by one (self.store_to_ram==False), where only 1 cycle was measured (len(self.arrival_times)==1)
        #2) Measured ncy cycles one by one (self.store_to_ram==False), where more than 1 cycle was measured (len(self.arrival_times)>1)
        #3) Measured ncy cycles in 1 pack (self.store_to_ram==True), where we only 1 cycle was measured (len(self.arrival_times)==1)
        #4) Measured ncy cycles in 1 pack (self.store_to_ram==True), where more than 1 cycle was measured (len(self.arrival_times)>1)

        if not self.store_to_ram: #Measureed ncy cycles one by one
            if len(self.arrival_times)==1: #Only one cycle was measured
                case=1 #we do not have enough data to calculate a median, so we use the mean value.
                cdt_median=cdt_mean #ms/cy
            else:
                case=2
                #We have a data arrival time for each cycle, so we can calculate the deltas between them.
                #By analyzing the deltas in the data arrival times, we can discard eventual outliers, (cycles arriving later than expected, eventually).
                #So we can calculate a more robust "future cycle delay time" by using the median of the data arrival time deltas, minus the integration time.
                deltas=1000.0*np.diff(self.arrival_times) #s->ms #delta of data arrival events (ddae)
                deltas_mean=np.mean(deltas) #mean of the deltas
                deltas_median=np.median(deltas) #median of the deltas
                deltas_max=np.max(deltas) #max of the deltas
                deltas_min=np.min(deltas) #min of the deltas
                stdev=np.std(deltas) #stdev of the deltas
                cdt_median=deltas_median-self.it_ms #"median" cycle delay time
                #we use the median instead of the mean to avoid eventual outliers.
        else: #Measured ncy cycles in 1 pack (store to ram)
            if len(self.arrival_times)==1: #Only one cycle was measured
                case=3 #we do not have enough data to calculate a median, so we use the mean value.
                cdt_median=cdt_mean #ms/cy
            else: #More than one cycle was measured
                #In this case, we do have a data arrival time for each cycle, but there will be as many repeated
                #values as times the callback function was called, to notify that X cycles were ready to be read.
                #So we cannot use these repeated timestamps to calculate the deltas.
                case=4 #Since we do not have enough data to calculate a median, we use the mean value.
                cdt_median=cdt_mean #ms/cy

        #Show results:
        if showinfo:
            self.logger.info("---Spec "+self.alias+" last measurement stats:---")
            self.logger.info("measured "+str(self.ncy_read) + " cycles at IT="+str(self.it_ms)+" ms")
            self.logger.info("Real_dur(meas)="+str(real_dur_meas)+"ms, Real_dur(final data handling)="+str(real_dur_fdh)+"ms, Real_dur(total)="+str(real_dur_total)+"ms")
            self.logger.info("Expected_min_dur(meas) (=ncy*IT) = "+str(expected_min_dur_meas)+"ms")
            self.logger.info("Measurement_delay = Real_dur(meas) - Expected_min_dur(meas) = "+str(real_dur_meas - expected_min_dur_meas)+"ms")
            self.logger.info("Mean cycle delay time: cdt_mean=max(0,Measurement_delay/ncy) = "+str(cdt_mean)+"ms/cy")
            if case in [1,3,4]:
                self.logger.info("The last measurement cycle delay time has been determined as (=cdt_mean)="+str(cdt_median)+"ms/cy")
            elif case == 2:
                self.logger.info("delta of data arrival events (ddae), last 5: "+str(deltas[-5:])+" ms") #last 5
                self.logger.info("delta of data arrival events: ddae_mean="+str(deltas_mean)+", ddae_median="+str(deltas_median)+", ddae_max="+str(deltas_max)+", ddae_min="+str(deltas_min)+", ddae_std="+str(stdev)+" [ms] (IT="+str(self.it_ms)+" ms)")
                self.logger.info("The last measurement cycle delay time has been determined as max(0,ddae_median("+str(deltas_median)+"ms)-IT("+str(self.it_ms)+"ms))="+str(cdt_median)+"ms/cy")
            self.logger.info("--------")

        return cdt_mean,cdt_median,real_dur_meas,real_dur_fdh,deltas_max,deltas_min

    #Auxiliary functions for recovery:

    def reset_device(self):
        """
        Perform a hard reset on the given spectrometer
        (microprocessor and USB controller will be reset).
        Note: The spec will start its reset procedure right after sending the command response back to the host.
        So it is needed to wait a bit after the response is received, to have the spec and usb properly initialized.
        Note: Only valid for newest spectrometers, AS7007, the AS7010 and AS-MINI. (Not AS5216)
        """
        self.logger.info("Resetting spectrometer device "+self.alias+"...")

        resdll=self.dll_handler.AVS_ResetDevice(self.spec_id)
        res=self.get_error(resdll)
        if res!="OK":
            res="Could not reset spectrometer "+self.alias+". Error: "+res
            self.logger.error("reset_device, "+res)
        else:
            sleep(5)

        self.error=res
        return res

    def test_recovery(self):
        """
        Test the recovery function of the spectrometer after a communication failure.

        This function will order a measurement, then it will propose the user to unplug and plug the spectrometer USB from the PC,
        and then it will try to recover the spectrometer communication.
        """
        self.logger.info("Testing recovery of spectrometer "+self.alias)

        #Set a large integration time
        res=self.set_it(it_ms=8000.0)

        #Measure -non blocking call-:
        if res=="OK":
            self.measure(ncy=1)
            self.docatch=False #-> Ignore new data arrivals on purpose.

        #Emulate a spectrometer connection/power loss
        if res=="OK":
            j=10
            while j>0:
                self.logger.info("------>  Unplug & Plug the spectrometer "+self.alias+" ("+self.sn+") USB from the PC now. ("+str(j)+"/10s) <------")
                j-=1
                sleep(1)

        #Start soft recovery:
        if res=="OK":
            res=self.recovery(dofree=False)
            #If regular recovery does not work, try with dofree:
            if res!="OK":
                res=self.recovery(dofree=True)

        #Finally, try to do a short testing measurement to check if the recovery was successful:
        if res=="OK":
            res=self.set_it(it_ms=100.0)
            if res=="OK":
                res=self.measure(ncy=1)
            if res=="OK":
                #wait for the measurement to be completed:
                res=self.wait_for_measurement()

        #Inform if the measurement was successful or not:
        if res=="OK":
            self.logger.info("Recovery test of spectrometer "+self.alias+" was successful.")
        else:
            self.logger.error("Recovery test of spectrometer "+self.alias+" failed. res="+res)

        return res






if __name__ == "__main__":
    print("Testing Avantes Spectrometer Class")

    #----Select Testing parameters----:
    #Configure DLL path (having into account if running python 32bits, or python 64bits):
    arch=platform.architecture()[0]
    if arch=="32bit":
        #dll_path = os.path.abspath("../../lib/oslib/spec_ava1/Avaspec-DLL_9.14.0.0_32bits/avaspec_v9.14.0.0.dll")
        dll_path = os.path.abspath("../../lib/oslib/spec_ava1/Avaspec-DLL_9.14.0.9_32bits/avaspec.dll")
        # dll_path = os.path.abspath("../../lib/oslib/spec_ava1/Avaspec-DLL_9.14.0.0_32bits/avaspec_v9.14.0.0.dll")
    else:
        # dll_path=os.path.abspath("../../lib/oslib/spec_ava1/Avaspec-DLL_9.14.0.0_64bits/avaspecx64_v9.14.0.0.dll")
        dll_path = os.path.abspath("../../lib/oslib/spec_ava1/Avaspec-DLL_9.14.0.9_64bits/avaspecx64.dll")
        

    #Select spectrometer device to be tested:
    #instr="P209_S1" #EVO CMOS (AVASpec-ULS2048CL-EVO-NGD6)
    #instr="P209_S2" #EVO CMOS (AvaSpec-ULS2048CL-EVO-SCI7)
    #instr="P121_S1"
    #instr="P998_S1" #CMOS ULS1024-USB2-FCPC
    #instr="P998_S2" #CCD 2048x14-SPU2-FC
    # instr = "test"

    # instruments=["P998_S1","P998_S2"]
    instruments=["test"]
    simulation_mode=[False,False]
    debug_mode=[3,3]
    store_to_ram=[False,False]
    dll_logging=[True,True] #internal dll logging file
    do_recovery_test=False
    do_performance_test=True
    performance_test_fpath="C:/Temp"

    instruments=instruments[:]

    #---------Testing Code Start Here-----

    SP=[]
    for i in range(len(instruments)):
        instr=instruments[i]

        #Initialize an Avantes spectrometer instance:
        SP.append(Avantes_Spectrometer())

        #Configure instance:
        SP[i].dll_path=dll_path
        SP[i].debug_mode=debug_mode[i]
        SP[i].store_to_ram=store_to_ram[i]
        SP[i].dll_logging=dll_logging[i]
        SP[i].simulation_mode=simulation_mode[i]
        SP[i].alias=str(i+1)

        if instr=="P209_S1":
            SP[i].sn="2106511U1" #CMOS_EVO
            SP[i].npix_active=2048
            SP[i].min_it_ms=0.03
            SP[i].performance_test_it_ms_list=np.arange(0.03,10.1,0.1)
            #SP[i].performance_test_ncy_list=[20 for _ in SP[i].performance_test_it_ms_list]
        elif instr=="P209_S2":
            SP[i].sn="2106510U1" #CMOS_EVO
            SP[i].npix_active=2048
            SP[i].min_it_ms=0.03
            SP[i].performance_test_it_ms_list=np.arange(0.03,10.1,0.1)
            #SP[i].performance_test_ncy_list=[20 for _ in SP[i].performance_test_it_ms_list]
        elif instr=="P121_S1":
            SP[i].sn="1510099U1"
            SP[i].npix_active=2048
            SP[i].min_it_ms=2.4
            #use the default performance_test_it_ms_list
        elif instr=="P121_S2":
            SP[i].sn="1511190U1"
            SP[i].npix_active=2048
            SP[i].min_it_ms=2.4
            #use the default performance_test_it_ms_list
        elif instr=="P998_S1":
            SP[i].sn="1102185U1"
            SP[i].npix_active=1024 #CMOS (OLD)
            SP[i].npix_blind_left=0
            SP[i].min_it_ms=2.4 #real one is 2.2
        elif instr=="P998_S2":
            SP[i].sn="0705002U1"
            SP[i].npix_active=2048 #CCD 2048x14-SPU2-FC
            SP[i].npix_blind_left=0 #real one is 8
            SP[i].min_it_ms=2.4 #real one is 2.17
        elif instr=="ibk_lab_1":
            SP[i].sn="1712086U1"
            SP[i].npix_active=2048
            SP[i].npix_blind_left=0
            SP[i].min_it_ms=3.0
        elif instr=="ibk_lab_2":
            SP[i].sn="1910091U1"
            SP[i].npix_active=2048
            SP[i].npix_blind_left=0
            SP[i].min_it_ms=3.0
        elif instr=="test":
            SP[i].sn="2203162U1"
            SP[i].npix_active=2048
            SP[i].npix_blind_left=0
            SP[i].min_it_ms=2.4
        else:
            raise("Instrument "+instr+" not recognized.")

        #Configure the logger for the spec alias:
        SP[i].initialize_spec_logger()

    #Configure logging:
    if np.any(debug_mode):
        loglevel=logging.DEBUG #TODO: Create independent loggers for each spectrometer
    else:
        loglevel=logging.INFO

    # logging formatter
    log_fmt="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(name)s] [%(message)s]"
    log_datefmt="%Y%m%dT%H%M%SZ"
    formatter = logging.Formatter(log_fmt, log_datefmt)

    # Configure the basic properties for logging
    logfile="C:/Temp/avantes_spec_test_"+"_".join(instruments)+".txt"
    print("logging into : "+logfile)
    logging.basicConfig(level=loglevel,format=log_fmt,datefmt=log_datefmt,
                        filename=logfile)

    # Create a StreamHandler for console output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(loglevel)  # Set the log level for the console handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger
    logging.getLogger().addHandler(console_handler)


    #---Testing actions---
    res="OK"

    #Connect the spectrometers:
    logger.info("--- Connecting spectrometers ---")
    for i in range(len(SP)):
        res=SP[i].connect()
        if res!="OK":
            break

    #Set integration time:
    if res=="OK":
        logger.info("--- Setting IT ---")
        for i in range(len(SP)):
            res=SP[i].set_it(it_ms=200.0)
            if res!="OK":
                break

    #Do testing Measurement:
    if res=="OK":
        logger.info("--- Starting measurements ---")
        for i in range(len(SP)):
            res=SP[i].measure(ncy=10)

    #Wait for measurement finished
    if res=="OK":
        for i in range(len(SP)):
            if SP[i].measuring:
                res=SP[i].wait_for_measurement()
        logger.info("All measurements done")

    logger.info("Waiting 5 seconds")
    sleep(5)

    if res=="OK" and do_recovery_test:

        #Test if recovery works:
        if res=="OK":
            sleep(3)
            logger.info("--- Testing if soft-recovery works in the last connected spec ---")
            res=SP[i].test_recovery()
            sleep(1)

        #Do a final measurement test after recovery, to ensure all specs are still working after recovery:
        #Set integration time:
        if res=="OK":
            logger.info("--- Final Test - Setting IT ---")
            for i in range(len(SP)):
                res=SP[i].set_it(it_ms=200.0)
                if res!="OK":
                    break

        #Do testing Measurement:
        if res=="OK":
            logger.info("--- Final Test - Starting measurements ---")
            for i in range(len(SP)):
                res=SP[i].measure(ncy=10)
                if res!="OK":
                    break

        #Wait for measurement finished
        if res=="OK":
            for i in range(len(SP)):
                if SP[i].measuring:
                    logger.info("--- Waiting for measurement of spectrometer "+SP[i].alias+" to finish... ---")
                    res=SP[i].wait_for_measurement()

    if res=="OK" and do_performance_test:
        for i in range(len(SP)):
            SP[i].performance_test(fpath=performance_test_fpath)
            logger.info("Performance test for spec "+str(i+1)+" finished.")

    #Finally, disconnect
    logger.info("--- Test Finished, disconnecting... ---")
    sleep(2)
    for i in range(len(SP)):
        res=SP[i].disconnect(dofree=True)

    logger.info("--- Finished ---")





