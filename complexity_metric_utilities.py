import re
import bisect

import pydicom
import glob 
import numpy as np
import pandas as pd

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks, shape_index
from scipy import ndimage as ndi
from skimage.feature import CENSURE


##
## Metrics
# Ported to python and independently modified from clinical 
# C# ESAPI-based complexity module originally written by Lukas Van Dyke

def calc_MLC_beam_complexity(plan_dataset,beam_num,f_val=1): 
    """ Calculates metric based on number of occurences of an mlc leaf moving with greater speed
        than std dev of all leaves for that arc. Normalized by count of control points 
       (with zero dose regions of the arc ignored).
       Simplified/modified form inspired by MI_s (Park 2014, Webb 2003)

        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
            f_val         : Multiplicative factor the standard deviation 
        Returns:
            total_count_weighted  : Normalized value mlc speed complexity
            standard_dev          : Standard deviation for *all* leaf movements during the arc 
            
    """
    MU_mat = normalize_peak_to_peak(get_MU_matrix(plan_dataset,beam_num)) # shift ~0 as 0
    mlc_speeds = get_all_mlc_speeds_matrix(plan_dataset,beam_num)
    standard_dev = f_val*np.std(mlc_speeds)
    num_of_zero_MU_control_points= np.count_nonzero(MU_mat==0)
    
    control_point_count = get_cp_count(plan_dataset,beam_num)-num_of_zero_MU_control_points  
    control_point_factor = 1/control_point_count

    count_over = (mlc_speeds > standard_dev).sum()
    total_count_weighted = count_over *control_point_factor

    return total_count_weighted,standard_dev
    
def calc_MLC_beam_acc_complexity(plan_dataset,beam_num,f_val=1): 
    """ Calculates metric based on number of occurences of an mlc leaf moving with greater acceleration
        than std dev of all leaves for that arc. Normalized by count of control points 
       (with zero dose regions of the arc ignored).
       Simplified/modified form inspired by MI_a (Park 2014, Webb 2003)

        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
            f_val         : Multiplicative factor the standard deviation 
        Returns:
            total_count_weighted  : Normalized value mlc speed complexity
            standard_dev          : Standard deviation for *all* leaf movements during the arc 
            
    """
    MU_mat = normalize_peak_to_peak(get_MU_matrix(plan_dataset,beam_num))
    mlc_acc = get_all_mlc_acc_matrix(plan_dataset,beam_num)
    standard_dev = f_val*np.std(mlc_acc)
    num_of_zero_MU_control_points= np.count_nonzero(MU_mat==0)
    
    control_point_count = get_cp_count(plan_dataset,beam_num)-num_of_zero_MU_control_points  
    control_point_factor = 1/control_point_count 

    count_over = (mlc_acc > standard_dev).sum()
    total_count_weighted = count_over *control_point_factor

    return total_count_weighted,standard_dev

def calc_unified_mlc(plan_dataset,beam_num,f_range=[0.2,0.5,1,2]):
    """ Calculates metric based on number of occurences of an mlc leaf moving with greater speed
        than std dev of all leaves for that arc. Normalized by count of control points 
       (with zero dose regions of the arc ignored. Summation is taken across range of f_values
       Alternative simplified/modified form inspired by MI_s (Park 2014, Webb 2003)

        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
            f_range         : Multiplicative factor the standard deviation, for summation
        Returns:
            score        : Summed result of f_range scores
            
    """
    res_list = []
    for f_val in f_range:
        res,_ = calc_MLC_beam_complexity(plan_dataset,beam_num,f_val)
        res_list.append(res)
    score = np.sum(np.array(res_list))
    return score

def calc_unified_mlc_acc(plan_dataset,beam_num,f_range=[0.2,0.5,1,2]):
    """ Calculates metric based on number of occurences of an mlc leaf moving with greater speed
        than std dev of all leaves for that arc. Normalized by count of control points 
       (with zero dose regions of the arc ignored. Summation is taken across range of f_values
       Alternative simplified/modified form inspired by MI_a (Park 2014, Webb 2003)

        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
            f_range         : Multiplicative factor the standard deviation, for summation
        Returns:
            score        : Summed result of f_range scores
            
    """
    res_list = []
    for f_val in f_range:
        res,_ = calc_MLC_beam_acc_complexity(plan_dataset,beam_num,f_val)
        res_list.append(res)
    score = np.sum(np.array(res_list))
    return score

def calc_area_metric(plan_dataset,beamnum,small_weight=1,big_weight=1):
        """ Calculates metric based on total area of open aperture  at a given control point,
            weighted by MU at that control point. Normalized by total beam MU. 
            Weight factor applied to area created by larger vs smaller mlc leaves 
            for Varian HD120 MLC bank (additional weighting, on top of scaling required based on leaf size)

        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beamnum      : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
            small_weight : weight applied to finer inner leaves. Unused in this study, default set to 1
            big_weight   : weight applied to larger outer leaves. Varied for purposes of study. 
        Returns:
            study_score  : Calculated area metric score for given big_weight
            
    """
    MU = get_total_beam_MU(plan_dataset,beamnum)
    cpcount = get_cp_count(plan_dataset,beamnum)
    score = 0
    for i in range(0,cpcount):
        muatcp = get_MU_at_CP(plan_dataset,beamnum,i)
        areaatcp = calc_control_point_aperture_area(plan_dataset,beamnum,i,small_weight,big_weight)
        res = np.round(muatcp * ( areaatcp),2)/MU
        score += res
    study_score = score/100 
    return study_score


def calc_gantry_speed_complexity_score(plan_dataset,beam_num, type="vel"):
    """ Calculates two types of complexity scores based on the gantry modulation: velocity or acceleration
        Each is an absolute sum of the values at each cp, zero dose regions ignored 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
            type         : String, either 'vel' or 'acc' depending on which score is desired. 
        Returns:
            gsscore       : Normalized value for the gantry speed modulation score 
    """
    controlpointtime = calc_treatment_times_via_gantry_speed_matrix(plan_dataset,beam_num)
    if np.isnan(controlpointtime).sum() > 0: #if any values are NaN, use other matrix. 
        print("NaN encountered in get_all_mlc_speeds_matrix")
        controlpointtime = calc_treatment_times_via_dose_rate_matrix(plan_dataset,beam_num)
    
    controlpointveltime = controlpointtime
    controlpointacctime = np.diff(controlpointveltime)
    gs_dpos_matrix = get_delta_gantry_angle_matrix(plan_dataset,beam_num)
    gs_ddpos_matrix = np.diff(gs_dpos_matrix)
    
    if type == "vel":
        gs_matrix = np.true_divide(gs_dpos_matrix,controlpointveltime)
    if type == "acc":
        gs_matrix = np.true_divide(gs_ddpos_matrix,controlpointacctime)
    
    gs_matrix[np.isnan(gs_matrix)] = 0
    gs_score = np.absolute(gs_matrix).sum()
    
    MU_mat = normalize_peak_to_peak(get_MU_matrix(plan_dataset,beam_num)) #normalized to set ~0 as 0
    num_of_zero_MU_control_points= np.count_nonzero(MU_mat==0)

    max_gantry = get_max_gantry_speed(plan_dataset)
    cp_count = get_cp_count(plan_dataset,beam_num)-num_of_zero_MU_control_points
    normalization_val = cp_count*max_gantry
    
    return gs_score/normalization_val
    
def calc_beam_aperture_complexity(plan_dataset,beamnum):
    """ Calculates an aperture complexity score based on (Younge 2012) (Carlsson 2008) which considers
        the ratio of MLC side edge length and aperture. Weights directly by MU at control point. 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beamnum     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            study_score  : Calculated aperture complexity score 
    """
    MU = get_total_beam_MU(plan_dataset,beamnum)
    cpcount = get_cp_count(plan_dataset,beamnum)
    score = 0
    for i in range(0,cpcount):
        muatcp = get_MU_at_CP(plan_dataset,beamnum,i)
        perimatcp = get_control_point_aperture_perimeter(plan_dataset,beamnum,i)
        areaatcp = calc_control_point_aperture_area(plan_dataset,beamnum,i)
        res = np.round(muatcp * ( perimatcp / areaatcp),5)/MU
        score += res

    study_score = score*100
    return study_score

def get_field_sizes(plan_dataset,beamno):
    """ Collects maximum field size in X and Y over entire arc. Validated via 
        comparison within ARIA Eclipse.
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beamno     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            fieldsizeX  : Field size in X (mm) 
            fieldsizeY  : Field size in Y (mm) 
    """
    testmax = []
    for i in range(0,get_cp_count(plan_dataset,beamno)):
        len_sequence = len(plan_dataset.BeamSequence[beamno].ControlPointSequence[i].BeamLimitingDevicePositionSequence)
        if len_sequence == 3:
            ASYM_X = np.array(plan_dataset.BeamSequence[beamno].ControlPointSequence[i].BeamLimitingDevicePositionSequence[0].LeafJawPositions)
            ASYM_Y = np.array(plan_dataset.BeamSequence[beamno].ControlPointSequence[i].BeamLimitingDevicePositionSequence[1].LeafJawPositions)
        testmax.append([ASYM_X[1],-ASYM_X[0],ASYM_Y[1],-ASYM_Y[0]])
    testmax= np.array(testmax)
    fieldsizeX = np.amax(testmax[:,0]) + np.amax(testmax[:,1])
    fieldsizeY = np.amax(testmax[:,2]) + np.amax(testmax[:,3])

    return fieldsizeX,fieldsizeY

##
## etc 
##

def flatten(l, ltypes=(list, tuple)):
    # via https://stackoverflow.com/questions/716477/join-list-of-lists-in-python
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)
    
def normalize_peak_to_peak(d):
    # d is a (n x dimension) np array
    d -= np.min(d, axis=0)
    d /= np.ptp(d, axis=0) #max-min
    return d

##
## General Purpose Beam Information
##

def get_total_beam_MU(plan_dataset,beam_num):
    """ Returns the total beam MU as a float pulled from the plan file. 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match. 
        Returns:
            totalMU      : Total MU planned for this beam 
    """
    totalMU = plan_dataset.FractionGroupSequence[0].ReferencedBeamSequence[beam_num].BeamMeterset
    return totalMU

def get_max_gantry_speed(plan_dataset):
    """ Uses hardcoded values via to return the maximum gantry speed for the unit found in the plan. 
        May need to be modified with new units being installed
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            
        Returns:
            maxgantryspeed      : Maximum gantry speed 
    """
    unitnum = get_unit_number(plan_dataset)
    maxgantryspeed = 0
    if unitnum == None:
        print("Non-integer unit number, MaxGantrySpeed assumed 6deg/sec")
        maxgantryspeed=6.0 # deg/sec
        return maxgantryspeed
    if unitnum == 10:
        maxgantryspeed=4.8 #
        return maxgantryspeed
    else:
        maxgantryspeed=6.0 #as above 
        return maxgantryspeed

def get_dose_rate_set(plan_dataset,beam_num):
    """ Returns the maximum dose rate for this plan. 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            drset        : Dose rate set for this plan
            
        Raises : 
            TypeError    : If the indexing is non-traditional, this will raise and invalidate the plan. 
    """
    drset = plan_dataset.BeamSequence[beam_num].ControlPointSequence[0].DoseRateSet
    try:
        drset = int(drset)
        return drset
    except TypeError:
        print("Non-integer returned for Dose Rate")
        return None    

def get_cp_count(plan_dataset,beam_num):
    """ Returns the number of control points for this beam. 
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            cpcount        : Integer value, number of control points 
    """
    cpcount = plan_dataset.BeamSequence[beam_num].NumberOfControlPoints
    return cpcount

def calc_SDT(dangle,dmu,maxgantryspeed,maxdoserate):
    """ Calculates the Segment Delivery Times (time between control points) using two methods, then finds maximum possible time. 
    
        Parameters:
            dangle         : Delta Angle matrix (change in gantry angles), via get_delta_gantry_angle_matrix(...) 
            dmu            : Delta MU matrix (change in delivered MU)    , via get_delta_MU_matrix(...)
            maxgantryspeed : Maximum gantry speed for this arc           , via get_max_gantry_speed(...)
            maxdoserate    : Dose rate set for this arc                  , via get_dose_rate_set(...)
            
         Returns: 
            maxTime        : A matrix of the segment delivery times, element-wise maximum possible time for each entry.  
    """
    #segment is portion between control points eh. dangle and dmu. 
    maxdoserate = maxdoserate/60 #convert to seconds      
    RotTimeViaMaxSpeed   = np.absolute(dangle / maxgantryspeed) 
    DelivTimeViaDoseRate = np.absolute(dmu   /  maxdoserate) 
    maxTime = np.around(np.maximum(RotTimeViaMaxSpeed,DelivTimeViaDoseRate),decimals=2)
    return maxTime

def get_lumped_leaf_positions(plan_dataset,beam_num,cpnum):
    """ Returns the MLC leaf positions directly pulled from the plan dataset. 
        Function can be used iteratively to information of entire arc.
        
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
            cpnum        : The control point currently being examined 
            
        Returns:
            lumpbank     : A numpy matrix containing all of the leaf positions for a given beam, given control point. 
                           Contains 2*N values, where N is the Number of Leaf/Jaw Pairs(element) subscript order 101, 102, … 1N, 201, 202, … 2N.
    """
    len_sequence = len(plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence)
    # if 1st entry in cp sequence else
    if len_sequence == 3:
        lumpbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[2].LeafJawPositions
        return lumpbank
    elif len_sequence == 1:
        lumpbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[0].LeafJawPositions
        return lumpbank
    else:
        return None
        
def get_unit_number(plan_dataset,beamnum=0):
    """ Utility function
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of any beam in plan, will be the same unit for any
        Returns:
            cpcount      : Integer value, number of control points 
    """
    namer = plan_dataset.BeamSequence[beamnum].TreatmentMachineName
    unitnum = namer[4] #hardcoding as field is standard naming:'unit5ser2899'
    try:
        unitnum = int(unitnum)
        return unitnum
    except TypeError:
        print("Non-integer returned for Unit Number")
        return None

def calc_dr_and_gs_at_cp_matrix(plan_dataset,beam_num):
    """ Calculates the Dose Rate and Gantry Speed matrices (used side by side later, so calculated together) 
             
    
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            doserate     : Matrix where each entry is the dose rate at the indexed control point, rounded to 2 sigfig
            gantryspeed  : Matrix where each entry is the gantry speed at the indexed control point, rounded to 2 sigfig
    """
    
    totalbeamMU = get_total_beam_MU(plan_dataset,beam_num)
    
    dangle = get_delta_gantry_angle_matrix(plan_dataset,beam_num)
    dmu = get_delta_MU_matrix(plan_dataset,beam_num)
    maxgs = get_max_gantry_speed(plan_dataset)  #degrees per second
    maxdr = get_dose_rate_set(plan_dataset,beam_num) # is in MU/min, /60 converts 
    segdeltime = calc_SDT(dangle,dmu,maxgs,maxdr)

    doserate = dmu/segdeltime *60
    gantryspeed = np.absolute(dangle/segdeltime)

    return np.around(doserate,decimals=2),np.around(gantryspeed,decimals=2)

def get_gantry_angle_matrix(plan_dataset,beam_num):
    """ Gets the gantry angle for all control points in arc  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset. Note: not always the same as 
                           filename, however, pdf_utilities.beam_sequence_init() ensures match.
        Returns:
            gangle       : Matrix of gantry angles where each entry is the gantry angle at indexed control point
    """
    gangle = [] # to build matrix for angles for all control points
    for i in range(0,get_cp_count(plan_dataset,beam_num)):
        gangle.append(plan_dataset.BeamSequence[beam_num].ControlPointSequence[i].GantryAngle)
    gangle = np.array(gangle) 
    return gangle

def get_delta_gantry_angle_matrix(plan_dataset,beam_num):
    """ Gets the change in gantry angle between control points in arc  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            dangle       : Matrix of gantry angles where each entry is the gantry angle between each indexed control point. 
                           for N control points, this has dimension N-1.  
    """
    gangle = get_gantry_angle_matrix(plan_dataset,beam_num)
    dangle = np.diff(gangle)
    dangle = np.where(dangle>180,360-dangle,dangle)
    dangle = np.absolute(dangle) 
    return dangle

def get_MU_matrix(plan_dataset,beam_num):
    """ Gets the MU matrix as calculated from Lukas' code for Aperture score 
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            MUcorrected  : Matrix of Meterset weighting at each control point against total beam MU
    """
    muraw = []
    totalBeamMU = get_total_beam_MU(plan_dataset,beam_num)
    for i in range(0,get_cp_count(plan_dataset,beam_num)):
        muraw.append(plan_dataset.BeamSequence[beam_num].ControlPointSequence[i].CumulativeMetersetWeight)
    mu = muraw
    MUcorrected = np.diff(np.array(mu)) * totalBeamMU 

    return MUcorrected

def get_delta_MU_matrix(plan_dataset,beam_num):
    """ Gets the change in MU for all control points in arc  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            dmu       : Matrix of gantry angles where each entry is the gantry angle between indexed control points
                        for N control points, this has dimension N-1.  
    """
    mumatrix = get_MU_matrix(plan_dataset,beam_num)
    dmu = np.diff(mumatrix)
    return dmu 

def get_number_of_beams(plan_dataset):
    """ Utility function: The number of beams in this plan file  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)

        Returns:
            beam_num     : Integer value, number of beams in plan.  
    """
    beam_num = len(plan_dataset.BeamSequence)
    return beam_num

def calc_treatment_times_via_gantry_speed_matrix(plan_dataset,beam_num):
    """ Gets the change in MU for all control points in arc  
        Parameters:
            plan_dataset : RP DICOM file for the plan being investigated, via pydicom.dcmread(filepath)
            beam_num     : Index of beam in plan_dataset, not always the same as 
                           filename however pdf_utilities.beam_sequence_init() is used to match.
        Returns:
            dmu       : Matrix of gantry angles where each entry is the gantry angle between indexed control points
                        for N control points, this has dimension N-1.  
    """
    dangle = get_delta_gantry_angle_matrix(plan_dataset,beam_num)
    dr, gs = calc_dr_and_gs_at_cp_matrix(plan_dataset,beam_num)
    treattime_gs = dangle/gs 
    return treattime_gs

def calc_treatment_times_via_dose_rate_matrix(plan_dataset,beam_num):    
    totalbeamMU = get_total_beam_MU(plan_dataset,beam_num)
    dmu = get_delta_MU_matrix(plan_dataset,beam_num)
    deliveredmu = dmu
    dr, gs = calc_dr_and_gs_at_cp_matrix(plan_dataset,beam_num)
    treatmentminutes = deliveredmu/dr
    treatmentsecondsmatrix = np.nan_to_num(treatmentminutes *60)
    return treatmentsecondsmatrix

def calc_total_time_from_treatment_times(times_matrix):
    res = np.sum(times_matrix)
    return res

def get_all_mlc_speeds_matrix(plan_dataset,beam_num):
    controlpointtime = calc_treatment_times_via_gantry_speed_matrix(plan_dataset,beam_num)
    if np.isnan(controlpointtime).sum() > 0: #if any values are NaN, use other matrix. 
        print("NaN encountered in get_all_mlc_speeds_matrix")
        controlpointtime = calc_treatment_times_via_dose_rate_matrix(plan_dataset,beam_num)
    
    leaf_pos_list = []
    for i in range(0,get_cp_count(plan_dataset,beam_num)):
        leaf_pos_list.append(get_lumped_leaf_positions(plan_dataset,beam_num,i))
        
    leaf_pos_arr = np.array(leaf_pos_list)
    leaf_diff_arr = np.absolute(np.diff(leaf_pos_arr,axis=0))
    #single_leafs = leaf_diff_arr.T #transposing for troubleshooting
    leaf_speeds = np.absolute(np.true_divide(leaf_diff_arr,controlpointtime[:,None]))
    return leaf_speeds

def get_all_mlc_acc_matrix(plan_dataset,beam_num):
    controlpointtime = calc_treatment_times_via_gantry_speed_matrix(plan_dataset,beam_num)
    if np.isnan(controlpointtime).sum() > 0: #if any values are NaN, use other matrix. 
        print("NaN encountered in get_all_mlc_speeds_matrix")
    controlpointacctime = controlpointtime[:-1]
    controlpointtime = calc_treatment_times_via_dose_rate_matrix(plan_dataset,beam_num)
    
    leaf_speed = get_all_mlc_speeds_matrix(plan_dataset,beam_num)

    leaf_speed_diff_arr =np.absolute(np.diff(leaf_speed,axis=0)) 

    leaf_acc = np.absolute(np.true_divide(leaf_speed_diff_arr,controlpointacctime[:,None]))
    
    return(leaf_acc)

def get_leaf_sizes(plan_dataset):
    unitnum = get_unit_number(plan_dataset)

    if (unitnum == 5 or unitnum == 3): 
        smallleaf, bigleaf, smallfromcenter = 2.5, 5, 16 #hardcoded
    else:
        smallleaf, bigleaf, smallfromcenter = 5, 10, 20  #hardcoded
    return smallleaf,bigleaf,smallfromcenter

def get_leaf_width(plan_dataset,currentleaf,totalleaves):
    if currentleaf < 1 or currentleaf > totalleaves :
        return None
    midpoint = totalleaves /2 
    sleaf, bleaf, smallfromcenter = get_leaf_sizes(plan_dataset) #smallfrom center is how many are small, counting out from center either side.
    if currentleaf > (midpoint-smallfromcenter) and currentleaf <= (midpoint +smallfromcenter):       
        return sleaf
    else:
        return bleaf 

def calc_leaf_width_matrix(plan_dataset,beamnum,cpnum,total_leaves_per_side,weight=1,big_weight=1):
    small,big,lcent = get_leaf_sizes(plan_dataset)
    mid = int(total_leaves_per_side/2) 
    matbank = get_leaf_positions_difference(plan_dataset,beamnum,cpnum,total_leaves_per_side)                     
    matbank[:(mid - lcent)] = matbank[:(mid - lcent)]*big*big_weight #large outer leaves
    matbank[(mid + lcent):] = matbank[(mid + lcent):]*big*big_weight #large outer leaves
    matbank[(mid - lcent):(mid + lcent)] = matbank[(mid - lcent):(mid + lcent)]*small*weight # small middle leaves
    return matbank 

def get_leaf_positions(plan_dataset,beam_num,cpnum,num_leaves=60):
    len_sequence = len(plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence)
    # if first of controlpointseq else
    if len_sequence == 1:
        leftbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[0].LeafJawPositions[:num_leaves]
        rightbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[0].LeafJawPositions[num_leaves:]
        return leftbank,rightbank    
    
    elif len_sequence == 3:    
        leftbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[2].LeafJawPositions[:num_leaves]
        rightbank = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cpnum].BeamLimitingDevicePositionSequence[2].LeafJawPositions[num_leaves:]
        return leftbank,rightbank

    else:
        return None

def get_leaf_positions_difference(plan_dataset,beam_num,cpnum,num_leaves):
    # to find open/closed gap areas in mlc leaves 
    leftbank, rightbank = get_leaf_positions(plan_dataset,beam_num,cpnum)
    rightmat = np.array(rightbank)
    leftmat = np.array(leftbank) 
    diffmat = leftmat-rightmat
    return diffmat

def calc_control_point_aperture_area(plan_dataset,beamno,cpnum,weight=1,big_weight=1): 
    adjusted_width_matrix = calc_leaf_width_matrix(plan_dataset,beamno,cpnum,60,weight,big_weight=1) 
    cpaperturearea = np.sum(adjusted_width_matrix)                                                 
    return cpaperturearea  
    
def get_control_point_aperture_perimeter(plan_dataset,beam_num,cpnum):
    lef, rig = get_leaf_positions(plan_dataset,beam_num,cpnum)
    
    lef = np.pad(lef,(1,1),'constant')
    rig = np.pad(rig,(1,1),'constant') #padding with zeros 
    
    lefdif, rightdif = np.diff(lef), np.diff(rig)
    prevPerim = np.sum(np.fabs(lefdif - rightdif))

    lefdifnext, rightdifnext = np.diff(np.flip(lef,0)), np.diff(np.flip(rig,0))
    nextPerim = np.sum(np.fabs(lefdifnext - rightdifnext))
    
    perim = prevPerim + nextPerim
    # Note:  np.diff returns out[n] = a[n+1] - a[n] 
    return perim

def get_MU_at_CP(plan_dataset,beam_num, cp_num):
    totalBeamMU = plan_dataset.FractionGroupSequence[0].ReferencedBeamSequence[beam_num].BeamMeterset
    if cp_num - 1 < 0:
        cp_num = 1 
    
    CulMetersetWeight = get_current_meterset_weight(plan_dataset,beam_num,cp_num)
    CulMetersetWeightPrev = get_current_meterset_weight(plan_dataset,beam_num,cp_num-1)
    MUatCP = totalBeamMU*(CulMetersetWeightPrev-CulMetersetWeight)
    return MUatCP

def get_current_meterset_weight(plan_dataset,beam_num,cp_num):
    culmeterweight = plan_dataset.BeamSequence[beam_num].ControlPointSequence[cp_num].CumulativeMetersetWeight
    return culmeterweight


    

