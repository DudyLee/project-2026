TOWER A | Data handoff for ML / LLM (CIVL 5220)
=================================================

THIS BUNDLE CONTAINS 2 DATASETS AND THIS REFERENCE NOTE.
1) SEND_ML_Hourly_07-18.csv: 175200 rows (10 Rooms x 4 PVSD angles x 4380 hours).
2) SEND_BIM_476Rooms.csv: geometry/solar context for 476 real Revit Rooms.

TIME CONVENTION:
All days, 07:00,08:00,...,18:00 INCLUSIVE. 12 hourly records/day (4380/year).
HOY_0based is 0..8759 in the ORIGINAL continuous 8760-h year; no weather year is asserted.
Month / Day / Hour and DOY_1based are retained for joining weather time-series.
All source 40 cases had the identical time index; no missing or duplicated hours.

HOURLY COLUMN DEFINITIONS:
Room_ID: Revit Room Number (text).
PVSD_Angle_deg: fixed louver angle for entire year; one of 0/30/60/90.
Sample_Role: TRAIN (7 rooms) or TEST (3 rooms). Never randomly split hours across rooms to claim external-room generalization.（3 TEST Rooms: BM532, BM702, BMG26. 7 TRAIN Rooms: BM223, BM432, BM441, BM522, BM608, LM608, LM703.）
Month, Day, Hour, DOY_1based, HOY_0based: source hourly timestamp.
Sensor_Count: sensors used in THAT room; varies across rooms, fixed across angles.
E_mean_lux: arithmetic average lux across all room grid sensors at this hour.
E_P10_lux: 10th percentile of room grid lux at this hour, numpy linear percentile.
F_300_fraction: fraction of grid sensors with illuminance >=300 lux in this hour (0..1).
F_UDI_fraction: fraction of grid sensors with 100 <= illuminance <=3000 lux in this hour (0..1).
PV_AC_kWh: modeled whole-system AC-generated electricity for that Room-case during this hour; kWh (NOT kW).
Hourly F_300 and F_UDI are SPATIAL fractions, NOT annual DA or UDI statistics.
PV residuals at -1e-19 kWh are clipped to zero; no daytime illuminance is modified.

BIM ROOM COLUMNS:
Room_ID joins SEND_ML_Hourly_07-18.csv Room_ID; 476 distinct rooms.
Level, Room_Area_m2, Window_Count, Window_Area_Nominal_m2, Revit_Window_Element_IDs: source BIM data.
WFR_window_to_floor: nominal window area/room floor area; NOT WWR.
Orientation_X_model/Y_model: room facing unit vector in local Rhino/Revit model axes, NOT geographic azimuth.
Glass_Area_m2: actual glass area from corrected glass mapping (original area unit conversion /10).
Annual_Incident_Shaded_kWh_m2 / Unshaded: glass-area weighted prior annual solar radiation results.
Solar_Obstruction_Ratio: 1 - shaded/unshaded, annual solar feature (0..1).
Sample_Role: only 10 rooms simulated, all other 466 rooms UNSIMULATED.