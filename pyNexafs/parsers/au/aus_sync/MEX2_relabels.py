"""
Dedicated file for MEX2 NEXAFS Relabels dictionary, due to large number of possible parameters.
"""

RELABELS = {
    ##################################
    ######################### MDA File
    ##################################
    "MEX2DCM01:ENERGY": "Energy Setpoint",
    "MEX2ES01ZEB01:CALC_ENERGY_EV": "Energy",
    "MEX2ES01ZEB01:GATE_TIME_SET": "Gate Time Setpoint",
    "MEX2SSCAN01:saveData_comment1": "Comment 1",
    "MEX2SSCAN01:saveData_comment2": "Comment 2",
    "MEX2ES01ZEB01:BRAGG_WITH_OFFSET": "Bragg",
    "SR11BCM01:CURRENT_MONITOR": "Current Monitor",  # What is this? I0?
    "MEX2ES01DAQ01:ch1:S:MeanValue_RBV": "Beam Intensity Monitor",  # Beam Intensity Monitor
    "MEX2ES01DAQ01:ch2:S:MeanValue_RBV": "I0",
    "MEX2ES01DAQ01:ch3:S:MeanValue_RBV": "SampleDrain",
    # 'MEX2ES01DAQ01:ch4:S:MeanValue_RBV',
    ("MEX2ES01DPP01:dppAVG:InputCountRate", "ICR_AVG"): "Input Average Count Rate",
    ("MEX2ES01DPP01:dppAVG:OutputCountRate", "OCR_AVG"): "Output Average Count Rate",
    # 'MEX2ES01DPP01:R:AVG:kCPS',
    "MEX2ES01DPP01:dppAVG:ElapsedLiveTime": "Count Time",
    # 'MEX2ES01ZEB01:PC_GATE_WID:RBV',
    # 'MEX2ES01DAQ01:ArrayCounter_RBV',
    # 'MEX2ES01DPP01:R:1:Total_RBV',
    # 'MEX2ES01DPP01:R:2:Total_RBV',
    # 'MEX2ES01DPP01:R:3:Total_RBV',
    # 'MEX2ES01DPP01:R:4:Total_RBV',
    # 'MEX2ES01DPP01:dpp1:InputCountRate',
    # 'MEX2ES01DPP01:dpp2:InputCountRate',
    # 'MEX2ES01DPP01:dpp3:InputCountRate',
    # 'MEX2ES01DPP01:dpp4:InputCountRate',
    # 'MEX2ES01DPP01:dpp1:OutputCountRate',
    # 'MEX2ES01DPP01:dpp2:OutputCountRate',
    # 'MEX2ES01DPP01:dpp3:OutputCountRate',
    # 'MEX2ES01DPP01:dpp4:OutputCountRate',
    # 'MEX2ES01DPP01:dpp1:ElapsedRealTime',
    # 'MEX2ES01DPP01:dpp2:ElapsedRealTime',
    # 'MEX2ES01DPP01:dpp3:ElapsedRealTime',
    # 'MEX2ES01DPP01:dpp4:ElapsedRealTime',
    # 'MEX2ES01DPP01:dpp1:ElapsedLiveTime',
    # 'MEX2ES01DPP01:dpp2:ElapsedLiveTime',
    # 'MEX2ES01DPP01:dpp3:ElapsedLiveTime',
    # 'MEX2ES01DPP01:dpp4:ElapsedLiveTime',
    # 'MEX2ES01DPP01:dpp1:DeadTime',
    # 'MEX2ES01DPP01:dpp2:DeadTime',
    # 'MEX2ES01DPP01:dpp3:DeadTime',
    # 'MEX2ES01DPP01:dpp4:DeadTime',
    # 'MEX2ES01DPP01:dpp1:PileUp',
    # 'MEX2ES01DPP01:dpp2:PileUp',
    # 'MEX2ES01DPP01:dpp3:PileUp',
    # 'MEX2ES01DPP01:dpp4:PileUp',
    # 'MEX2ES01DPP01:dpp1:F1PileUp',
    # 'MEX2ES01DPP01:dpp2:F1PileUp',
    # 'MEX2ES01DPP01:dpp3:F1PileUp',
    # 'MEX2ES01DPP01:dpp4:F1PileUp',
    # 'MEX2ES01DPP01:dpp1:Triggers',
    # 'MEX2ES01DPP01:dpp2:Triggers',
    # 'MEX2ES01DPP01:dpp3:Triggers',
    # 'MEX2ES01DPP01:dpp4:Triggers',
    # 'MEX2ES01DPP01:dpp1:Events',
    # 'MEX2ES01DPP01:dpp2:Events',
    # 'MEX2ES01DPP01:dpp3:Events',
    # 'MEX2ES01DPP01:dpp4:Events',
    # 'MEX2ES01DPP01:dpp1:F1DeadTime',
    # 'MEX2ES01DPP01:dpp2:F1DeadTime',
    # 'MEX2ES01DPP01:dpp3:F1DeadTime',
    # 'MEX2ES01DPP01:dpp4:F1DeadTime',
    # 'MEX2ES01DPP01:dpp1:FastDeadTime',
    # 'MEX2ES01DPP01:dpp2:FastDeadTime',
    # 'MEX2ES01DPP01:dpp3:FastDeadTime',
    # 'MEX2ES01DPP01:dpp4:FastDeadTime',
    # 'MEX2ES01DPP01:dpp1:InUse',
    # 'MEX2ES01DPP01:dpp2:InUse',
    # 'MEX2ES01DPP01:dpp3:InUse',
    # 'MEX2ES01DPP01:dpp4:InUse',
    # 'MEX2ES01DPP01:dpp:ArrayCounter_RBV',
    # MEX2SSCAN01:saveData_realTime1D
    # MEX2SSCAN01:saveData_fileSystem
    # MEX2SSCAN01:saveData_subDir
    # MEX2SSCAN01:saveData_fileName
    # MEX2SSCAN01:scan1.P1SM
    # MEX2SSCAN01:scan1.P2SM
    # MEX2SSCAN01:scan1.P3SM
    # MEX2SSCAN01:scan1.P4SM
    # MEX2SSCAN01:scanTypeSpec
    # MEX2SSCAN01:scan1.BSPV
    # MEX2SSCAN01:scan1.BSCD
    # MEX2SSCAN01:scan1.BSWAIT
    # MEX2SSCAN01:scan1.ASPV
    # MEX2SSCAN01:scan1.ASCD
    # MEX2SSCAN01:scan1.ASWAIT
    # MEX2SSCAN01:scan1.PDLY
    # MEX2SSCAN01:scan1.DDLY
    # MEX1ES01GLU01:MEX_TIME
    # MEX2MIR01MOT01.RBV
    # MEX2MIR01MOT02.RBV
    # MEX2MIR01MOT03.RBV
    # MEX2MIR01MOT04.RBV
    # MEX2MIR01MOT09.RBV
    # MEX2MIR01MOT10.RBV
    # MEX2MIR01MOT11.RBV
    # MEX2MIR01MOT12.RBV
    # MEX2FE01MIR01:X.RBV
    # MEX2FE01MIR01:Y.RBV
    "MEX2FE01MIR01:PITCH.RBV": "PITCH",
    "MEX2FE01MIR01:YAW.RBV": "YAW",
    # MEX2FE01MIR01:ENC_X.RBV
    # MEX2FE01MIR01:ENC_Y.RBV
    # MEX2FE01MIR01:ENC_PITCH.RBV
    # MEX2FE01MIR01:ENC_YAW.RBV
    # MEX2SLT01MOT01.RBV
    # MEX2SLT01MOT02.RBV
    # MEX2SLT01MOT03.RBV
    # MEX2SLT01MOT04.RBV
    # MEX2SLT01:VSIZE.RBV
    # MEX2SLT01:VCENTRE.RBV
    # MEX2SLT01:HSIZE.RBV
    # MEX2SLT01:HCENTRE.RBV
    # MEX2SLT01:VSIZE.OFF
    # MEX2SLT01:VCENTRE.OFF
    # MEX2SLT01:HSIZE.OFF
    # MEX2SLT01:HCENTRE.OFF
    # MEX2MIR02MOT01.RBV
    # MEX2MIR02MOT02.RBV
    # MEX2MIR02MOT03.RBV
    # MEX2MIR02MOT04.RBV
    # MEX2MIR02MOT05.RBV
    # MEX2MIR02:TRANS.RBV
    "MEX2MIR02:PITCH.RBV": "PITCH2",
    # MEX2MIR02:HEIGHT.RBV
    # MEX2MIR02:ROLL.RBV
    "MEX2MIR02:YAW.RBV": "YAW2",
    # MEX2MIR02TES04:TEMPERATURE_MONITOR
    # MEX2SLT02MOT01.RBV
    # MEX2SLT02MOT02.RBV
    # MEX2SLT02MOT03.RBV
    # MEX2SLT02MOT04.RBV
    # MEX2SLT02:VSIZE.RBV
    # MEX2SLT02:VCENTRE.RBV
    # MEX2SLT02:HSIZE.RBV
    # MEX2SLT02:HCENTRE.RBV
    # MEX2SLT02:VSIZE.OFF
    # MEX2SLT02:VCENTRE.OFF
    # MEX2SLT02:HSIZE.OFF
    # MEX2SLT02:HCENTRE.OFF
    # MEX2DCM01:ENERGY_RBV
    # MEX2DCM01:ENERGY_EV_RBV
    # MEX2DCM01:OFFSET_RBV
    # MEX2DCM01:XTAL_INBEAM.RVAL
    # MEX2DCM01:FINE_PITCH_MRAD_RBV
    # MEX2DCM01:FINE_ROLL_MRAD_RBV
    # MEX2DCM01MOT01.RBV
    # MEX2DCM01MOT02.RBV
    # MEX2DCM01MOT05.RBV
    # MEX2DCM01MOT03.RBV
    # MEX2DCM01MOT04.RBV
    # MEX2DCM01MOT01.OFF
    # MEX2DCM01MOT02.OFF
    # MEX2DCM01:y2_track
    # MEX2DCM01:y2_mvmin
    # MEX2DCM01:th_mvmin
    # MEX2DCM01:Dspace
    # MEX2DCM01:Mono111DSpace
    # MEX2DCM01:Mono111ThetaOffset
    # MEX2DCM01:Mono111HeightOffset
    # MEX2DCM01:Mono111Pitch
    # MEX2DCM01:Mono111Roll
    # MEX2DCM01:Mono111Centre
    # MEX2DCM01:MonoInSbDSpace
    # MEX2DCM01:MonoInSbThetaOffset
    # MEX2DCM01:MonoInSbHeightOffset
    # MEX2DCM01:MonoInSbPitch
    # MEX2DCM01:MonoInSbRoll
    # MEX2DCM01:MonoInSbCentre
    # MEX2AUTOROCK:PITCH_SCAN.P1WD
    # MEX2AUTOROCK:PITCH_SCAN.P1SI
    # MEX2AUTOROCK:PITCH_SCAN.NPTS
    # MEX2AUTOROCK:DETECTOR_SELECT
    # MEX2AUTOROCK:COUNTER
    # MEX2BIM01MOT01.RBV
    # MEX2BIM01:FOIL:select
    # MEX2BIM01AMP01:sens_put
    # MEX2BIM01AMP01:offset_put
    # MEX2BIM01AMP01:offset_on
    # MEX2BIM01AMP01:invert_on
    # MEX2BIM01AMP01:filter_type.RVAL
    # MEX2BIM01AMP01:low_freq.RVAL
    # MEX2REF01MOT01.RBV
    # MEX2REF01:REF:select
    # MEX2REF01AMP01:sens_put
    # MEX2REF01AMP01:offset_put
    # MEX2REF01AMP01:offset_on
    # MEX2REF01AMP01:invert_on
    # MEX2REF01AMP01:filter_type.RVAL
    # MEX2REF01AMP01:low_freq.RVAL
    # MEX2SLT03MOT01.RBV
    # MEX2SLT03MOT02.RBV
    # MEX2SLT03MOT03.RBV
    # MEX2SLT03MOT04.RBV
    # MEX2SLT03:VSIZE.RBV
    # MEX2SLT03:VCENTRE.RBV
    # MEX2SLT03:HSIZE.RBV
    # MEX2SLT03:HCENTRE.RBV
    # MEX2SLT03:VSIZE.OFF
    # MEX2SLT03:VCENTRE.OFF
    # MEX2SLT03:HSIZE.OFF
    # MEX2SLT03:HCENTRE.OFF
    "MEX2STG01MOT01.RBV": "Sample X",
    "MEX2STG01MOT02.RBV": "Sample Y",
    "MEX2STG01MOT03.RBV": "Sample Z",
    # MEX2STG01:XHAT.RBV
    # MEX2STG01:ZHAT.RBV
    # MEX2ES01MOT01.RBV
    # MEX2ES01AMP01:sens_put
    # MEX2ES01AMP01:offset_put
    # MEX2ES01AMP01:offset_on
    # MEX2ES01AMP01:invert_on
    # MEX2ES01AMP01:filter_type.RVAL
    # MEX2ES01AMP01:low_freq.RVAL
    # MEX2ES01AMP02:sens_put
    # MEX2ES01AMP02:offset_put
    # MEX2ES01AMP02:offset_on
    # MEX2ES01AMP02:invert_on
    # MEX2ES01AMP02:filter_type.RVAL
    # MEX2ES01AMP02:low_freq.RVAL
    # MEX2BLSH01SHT01:OPEN_CLOSE_STATUS
    # MEX2SZ03KSW01:KEY_CACHE_STATUS
    # MEX2SZ03KSW02:KEY_CACHE_STATUS
    # MEX2ES01DPP01:mca1.R0LO
    # MEX2ES01DPP01:mca1.R0HI
    # MEX2ES01DPP01:mca2.R0LO
    # MEX2ES01DPP01:mca2.R0HI
    # MEX2ES01DPP01:mca3.R0LO
    # MEX2ES01DPP01:mca3.R0HI
    # MEX2ES01DPP01:mca4.R0LO
    # MEX2ES01DPP01:mca4.R0HI
    # MEX2ES01DPP01:dpp1:InUse
    # MEX2ES01DPP01:dpp2:InUse
    # MEX2ES01DPP01:dpp3:InUse
    # MEX2ES01DPP01:dpp4:InUse
    # MEX2SSCAN01:PHASE_BOUNDARY_VALUE
    # MEX2SSCAN01:PHASE_STEP_VALUE
    # MEX2SSCAN01:PHASE_DURATION_VALUE
    # MEX2SSCAN01:PHASE_NUMBER_OF_POINTS
    # MEX2SSCAN01:PHASE_IN_USE
    # MEX2SSCAN01:PHASE_USES_KSPACE
    # MEX2SSCAN01:PHASE_USES_SQTIME
    # MEX2SSCAN01:TOTAL_NUMBER_OF_POINTS
    # MEX2SSCAN01:MODE
    # MEX2SSCAN01:EDGE_ENERGY
    "MEX2SSCAN01:SIMPLE_START_1_VALUE": "E1",
    # MEX2SSCAN01:SIMPLE_STEP_1_VALUE
    "MEX2SSCAN01:SIMPLE_END_1_VALUE": "E2",
    "MEX2SSCAN01:SIMPLE_START_2_VALUE": "E3",
    # MEX2SSCAN01:SIMPLE_STEP_2_VALUE
    "MEX2SSCAN01:SIMPLE_END_2_VALUE": "E4",
    # MEX2SSCAN01:SIMPLE_DURATION_VALUE
    # MEX2SSCAN01:SIMPLE_NUMBER_OF_POINTS
    (
        "MEX2ES01DPP01:ch1:W:ArrayData",
        "MEX2ES01DPP02:MCA1:ArrayData",
    ): "Fluorescence Detector 1",
    (
        "MEX2ES01DPP01:ch2:W:ArrayData",
        "MEX2ES01DPP02:MCA2:ArrayData",
    ): "Fluorescence Detector 2",
    (
        "MEX2ES01DPP01:ch3:W:ArrayData",
        "MEX2ES01DPP02:MCA3:ArrayData",
    ): "Fluorescence Detector 3",
    (
        "MEX2ES01DPP01:ch4:W:ArrayData",
        "MEX2ES01DPP02:MCA4:ArrayData",
    ): "Fluorescence Detector 4",
    ##################################
    ######################### ASC File
    ##################################
    "mda_version": "MDA File Version",
    "mda_scan_number": "Scan Number",
    "mda_rank": "Overall scan dimension",
    "mda_dimensions": "Total requested scan size",
    ##################################
    ######################### XDR File
    ##################################
    # "energy": "Energy",
    # "bragg": "Bragg",
    # "count_time": "Count Time",
    # "BIM": "BIM",
    # "i0": "IO",
    # "SampleDrain": "Sample Drain",
    #### XDI Names ####
    "ROI_AD_AVG": "ROI Average",
    "ifluor": "Fluorescence",
    "ROI.start_bin": "Fluorescence Start Bin",
    "ROI.end_bin": "Fluorescence End Bin",
    "Element.symbol": "Element",
    "Element.edge": "Absorption Edge",
}
