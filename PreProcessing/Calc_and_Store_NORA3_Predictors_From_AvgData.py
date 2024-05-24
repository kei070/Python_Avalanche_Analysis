"""
NOTE:
This script will only generate the predictors for those days for which we have an avalanche assessment!

Calculate and store the avalanche predictor data for a specific NORA3 gridcell.
The ideas for the predictors are taken from Gauthier et al. (2017) and Hendrikx et al. (2014), but note that not all of
their predictors are used and some definitions are slightly different.
"""


#%% imports
import os
import sys
import glob
import numpy as np
import geopandas as gpd
import pandas as pd
from netCDF4 import Dataset


#%% set scripts directory and import self-written functions
script_direc = "/PATH_TO_SCRIPTS/Python_Avalanche_Analysis/"
os.chdir(script_direc)
# needed for console
sys.path.append(script_direc)

from Functions.General_Functions.Func_Progressbar import print_progress_bar


#%% set region code
# reg_code = 3009  # Nord-Troms
# reg_code = 3010  # Lyngen
# reg_code = 3011  # Tromsoe
# reg_code = 3012  # Soer-Troms
# reg_code = 3013  # Indre Troms

try:
    reg_code = int(sys.argv[1])
except:
    reg_code = 3009
# end try except


#%% set low and high elevation threshold
try:
    h_low = int(sys.argv[2])
    h_hi = int(sys.argv[3])
except:
    h_low = 400
    h_hi = 900
# end try except


#%% use western/eastern exposure condition?
west_exp = False
east_exp = False

if west_exp:
    expos = "w_expos"
    expos_add = "_WestExposed"
else:
    expos_add = ""
# end if
if east_exp:
    expos = "e_expos"
    expos_add = "_EastExposed"
else:
    expos_add = ""
# end if


#%% set region name according to region code
f_add = ""
if reg_code == 3009:
    region = "NordTroms"
elif reg_code == 3010:
    region = "Lyngen"
elif reg_code == 3011:
    region = "Tromsoe"
elif reg_code == 3012:
    region = "SoerTroms"
elif reg_code == 3013:
    region = "IndreTroms"
# end if elif

print(f"\nGenerate the predictors for the days with risk assessment for region {region}.\n")


#%% load the avalanche risk list with Pandas
f_path = "/PATH_TO_AVALANCHE_RISK_FILE/"
f_name = "Avalanche_Risk_List.csv"

ar_df = pd.read_csv(f_path + f_name)


#%% convert the date column to datetime format
ar_df["date"] = pd.to_datetime(ar_df["date"])


#%% extract the region according to the region code
ar_reg = ar_df[ar_df.region == reg_code]


#%% load a NORA3 file to get the grid
data_path = "/PATH_TO_FILE_CONTAINING_NORA3_GRID/"
f_name = "fc2021030100_003_fp.nc"
nc = Dataset(data_path + f_name)


#%% extract the projection
crs = nc.variables["projection_lambert"].proj4


#%% load the surface geopotential
sgz = np.squeeze(nc.variables["surface_geopotential"][:] / 9.81)

lons_all = nc.variables["longitude"][:]
lats_all = nc.variables["latitude"][:]


#%% extract Northern Norway just by eye-balling it
# x1, x2 = 390, 500
# y1, y2 = 850, 950

# nnor_sgz = sgz[y1:y2, x1:x2]


#%% load lat and lon
# lons_nn = nc.variables["longitude"][y1:y2, x1:x2]
# lats_nn = nc.variables["latitude"][y1:y2, x1:x2]


#%% generate a hillshade elevation map
# hillshade = es.hillshade(sgz, azimuth=270, altitude=1)


#%% load the shape file with the surface height and exposure
shp_path = f"/PATH_TO_CELL_COORDINATES/Cells_Between_Thres_Height/NorthernNorway_Subset/{region}/"
shp = gpd.read_file(shp_path + f"NORA3_between{h_low}_and_{h_hi}m_Coords_in_{region}_Shape.shp")

# check if there are grid cells in the chosen altitude band
if len(shp) == 0:
    sys.exit(f"No gridcells for {reg_code} between {h_low} and {h_hi} m. Stopping execution.")
# end if


#%% select cells with western/eastern exposure
if (west_exp | east_exp):
    expos_cells = shp[shp[expos] == 1]

    exp_lat = np.array(expos_cells.lat)
    exp_lon = np.array(expos_cells.lon)
# end if


#%% NORA3 data path
data_path_atm = f"PATH_TO_NORA3_DATA/{region}/Between{h_low}_and_{h_hi}m/" + f_add


#%% load data
fns_atm = sorted(glob.glob(data_path_atm + "NORA3_*.csv"), key=str.casefold)

print(f"Number of files: {len(fns_atm)}.\n")
# print(f"Number of wind files: {len(fns_win)}.\n")


#%% extract all lats and lons from atm and wind files
lons_atm, lats_atm = [], []
lo_la_atm = []

count = 0
l = len(fns_atm)
print(f"Loading lats and lons ({l} iterations)...\n")
print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
for fn_atm in fns_atm:

    # load the files
    df_atm_temp = pd.read_csv(fn_atm, header=9)

    # get the coordinates of the NORA3 file from the file name
    lon = fn_atm.split("/")[-1].split("_")[1][3:]
    lat = fn_atm.split("/")[-1].split("_")[2][3:-4]
    # lons_atm.append(lon)
    # lats_atm.append(lat)

    lo_la_atm.append(f"{lon},{lat}")

    # print(f"{count}/{len(fns_atm)}")
    count += 1
    print_progress_bar(count, l, prefix='Progress:', suffix='Complete', length=50)
# end for fn_atm

lo_la_atm = np.array(lo_la_atm)

print(f"\nSet of {count} lats and lon loaded.\n")


#%% loop over the files and load the data

# set a boolean flag to check if there existed precipitation values > 10e7 and < 0
pr_large = False
pr_neg = False

lons = []
lats = []

w10m = []
wdir = []
at2m = []
prec = []
snow = []
rain = []

count = 0
l = len(lo_la_atm)
print(f"Loading NORA3 data ({l} iterations)...\n")
print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
for lo_la in lo_la_atm:
    # print(f"{count}/{len(lo_la_atm)}")

    # use only western-exposed cells?
    if (west_exp | east_exp):
       cond = ((float(lon) in exp_lon) & (float(lat) in exp_lat))
    else:
        cond = True
    # end if else

    lon, lat = lo_la.split(",")

    if cond:

        df_atm_temp = pd.read_csv(glob.glob(data_path_atm + f"/NORA3_lon{lon}_lat{lat}.csv")[0])

        # get the coordinates of the NORA3 file from the file name
        lons.append(float(lon))
        lats.append(float(lat))

        # extract the wind, temperature, and precipitation columns
        w10m.append(df_atm_temp["wind_speed"])
        wdir.append(df_atm_temp["wind_direction"])
        at2m.append(df_atm_temp["air_temperature_2m"])
        if any(df_atm_temp['precipitation_amount_hourly'] > 1e10):
            # print("\nprecip values >1e10 exist; adjusting to zero...")
            n_nan = np.sum(df_atm_temp['precipitation_amount_hourly'] > 1e10)
            perc = n_nan / len(df_atm_temp) * 100
            # print(f"{perc}% of precipitation values are >1e10")
            df_atm_temp.loc[df_atm_temp['precipitation_amount_hourly'] > 1e10, 'precipitation_amount_hourly'] = 0
            pr_large = True
        # end if
        if any(df_atm_temp['precipitation_amount_hourly'] < 0):
            # print("\nprecip values <0 exist; adjusting to zero...")
            n_nan = np.sum(df_atm_temp['precipitation_amount_hourly'] < 0)
            perc = n_nan / len(df_atm_temp) * 100
            # print(f"{perc}% of precipitation values are < 0")
            # print(f"largest magnitude: {df_atm_temp['precipitation_amount_hourly'].min()}\n")
            df_atm_temp.loc[df_atm_temp['precipitation_amount_hourly'] < 0, 'precipitation_amount_hourly'] = 0
            pr_neg = True
        # end if
        prec.append(df_atm_temp['precipitation_amount_hourly'])

        # generate a column with solid and a column with liquid precipitation: solid if temp < 0 degC, liquid if
        # temp > 0 degC this should be more precise than in Gauthier et al. (2017), who use a similar separation, but on
        # a daily level
        solid_prec = np.zeros(len(df_atm_temp))
        liquid_prec = np.zeros(len(df_atm_temp))
        solid_prec[df_atm_temp["air_temperature_2m"] < 273.15] = \
                                  df_atm_temp["precipitation_amount_hourly"][df_atm_temp["air_temperature_2m"] < 273.15]
        liquid_prec[df_atm_temp["air_temperature_2m"] > 273.15] = \
                                  df_atm_temp["precipitation_amount_hourly"][df_atm_temp["air_temperature_2m"] > 273.15]

        snow.append(solid_prec)
        rain.append(liquid_prec)

    else:
        continue
    # end if else
    count += 1
    print_progress_bar(count, l, prefix='Progress:', suffix='Complete', length=50)
# end for fn_atm, fn_win

print(f"\nWent through {count} iterations.\n")

if pr_large:
    print("There were precipitation values > 1e10. They were set to 0.")
if pr_neg:
    print("There were precipitation values < 0. They were set to 0.")
# end if
print("")


#%% convert the lat and lon lists to a dataframe to later store it as metadata for the predictors
lon_lat = pd.DataFrame({"lon":lons, "lat":lats})


#%% average the value across grid cells
w10m = np.mean(np.stack(w10m), axis=0)
wdir = np.mean(np.stack(wdir), axis=0)
at2m = np.mean(np.stack(at2m), axis=0)
prec = np.mean(np.stack(prec), axis=0)
snow = np.mean(np.stack(snow), axis=0)
rain = np.mean(np.stack(rain), axis=0)


#%% set up the hourly dataframe with the multi-gridcell averaged values
df_atm = pd.DataFrame({"date":pd.to_datetime(df_atm_temp.date),
                       "air_temperature_2m":at2m, "wind_speed_10m":w10m, "wind_direction":wdir,
                       "precipitation_amount_hourly":prec,
                       "solid_precip":snow, "liquid_precip":rain, "drift":w10m*prec, "drift3":w10m**3*prec})

df_atm.set_index("date", inplace=True)


#%% aggregate daily and calculate several quantities: mean, max, min, range (i.e., max-min) of temperature;
#   precipitation sum (solid, liquid, total), mean wind speed
df_atm_day = pd.DataFrame({"date":pd.to_datetime(df_atm.groupby(df_atm.index.date).mean().index),
                           "t_mean":df_atm["air_temperature_2m"].groupby(df_atm.index.date).mean(),
                           "t_max":df_atm["air_temperature_2m"].groupby(df_atm.index.date).max(),
                           "t_min":df_atm["air_temperature_2m"].groupby(df_atm.index.date).min(),
                           "t_range":df_atm["air_temperature_2m"].groupby(df_atm.index.date).max() -
                           df_atm["air_temperature_2m"].groupby(df_atm.index.date).min(),
                           "s1":df_atm["solid_precip"].groupby(df_atm.index.date).sum(),
                           "r1":df_atm["liquid_precip"].groupby(df_atm.index.date).sum(),
                           "total_prec_sum":df_atm["precipitation_amount_hourly"].groupby(df_atm.index.date).sum(),
                           "wspeed_mean":df_atm["wind_speed_10m"].groupby(df_atm.index.date).mean(),
                           "wspeed_max":df_atm["wind_speed_10m"].groupby(df_atm.index.date).max(),
                           "wind_direction":df_atm["wind_direction"].groupby(df_atm.index.date).mean()})

# add the freeze-thaw cycle (1 if yes, 0 of no)
ftc = np.zeros(len(df_atm_day))
ftc[((df_atm_day["t_min"] < 273.15) & (df_atm_day["t_max"] > 273.15))] = 1
df_atm_day["ftc"] = ftc

df_atm_day.set_index(pd.to_datetime(df_atm_day.index), inplace=True)


#%% extract the data for the dates with avalanche risk assessments
df_atm_risk = df_atm_day.merge(ar_reg, how="inner", on="date")


#%% set a day-of-year axis
df_atm_risk.set_index(pd.to_datetime(df_atm_risk.date), inplace=True)
df_atm_risk["DoY"] = df_atm_risk.index.dayofyear


#%% loop over the events and calculate also the accrued quantities
r2, r3, r4, r5, r6, r7 = [], [], [], [], [], []  # liquid precip accrued 2-7 days
s2, s3, s4, s5, s6, s7 = [], [], [], [], [], []  # solid precip accrued 2-7 days

dtemp1, dtemp2, dtemp3 = [], [], []  # temperature amplitude 1-3 days before an event
dtempd1, dtempd2, dtempd3 = [], [], []  # temperature amplitude between 1-3 days before and day of the event

t2, t3, t4, t5 = [], [], [], []  # average temperature 2-5 days before AND including the event

w2, w3, w4, w5 = [], [], [], []  # average wind speed 2-5 days before AND including the event

tmax2, tmax3, tmax4, tmax5 =  [], [], [], []  # average of max-temperature 2-5 days before AND including the event

wmax2, wmax3, wmax4, wmax5 =  [], [], [], []  # average of max-temperature 2-5 days before AND including the event

pdd = []  # positive degree-days

wdrift_2 = []  # wind drift = avg wind speed * precip; the day before and the day of the event
wdrift3_2 = []  # wind drift^3 = (avg wind speed)^3 * precip; the day before and the day of the event
wdrift_3 = []  # wind drift two days before up to the day of the event
wdrift3_3 = []  # wind drift^3 two days before up to the day of the event

avg_wspeed2 = []
avg_wspeed3 = []

count = 0
l = len(df_atm_risk.date)
print(f"Generating predictors for all days with risk assessment ({l} iterations)...\n")
print_progress_bar(0, l, prefix='Progress:', suffix='Complete', length=50)
for da in df_atm_risk.date:

    # find the index for the date of the event
    ind = np.where(df_atm_day.date == da)[0][0]

    # calculate the accrued liquid precipitation over multiple days
    r2.append(np.sum(df_atm_day["r1"].iloc[ind-1:ind+1]))
    r3.append(np.sum(df_atm_day["r1"].iloc[ind-2:ind+1]))
    r4.append(np.sum(df_atm_day["r1"].iloc[ind-3:ind+1]))
    r5.append(np.sum(df_atm_day["r1"].iloc[ind-4:ind+1]))
    r6.append(np.sum(df_atm_day["r1"].iloc[ind-5:ind+1]))
    r7.append(np.sum(df_atm_day["r1"].iloc[ind-6:ind+1]))

    # calculate the accrued solid precipitation over multiple days
    s2.append(np.sum(df_atm_day["s1"].iloc[ind-1:ind+1]))
    s3.append(np.sum(df_atm_day["s1"].iloc[ind-2:ind+1]))
    s4.append(np.sum(df_atm_day["s1"].iloc[ind-3:ind+1]))
    s5.append(np.sum(df_atm_day["s1"].iloc[ind-4:ind+1]))
    s6.append(np.sum(df_atm_day["s1"].iloc[ind-5:ind+1]))
    s7.append(np.sum(df_atm_day["s1"].iloc[ind-6:ind+1]))

    # calculate the average temperature over multiple days
    t2.append(np.mean(df_atm_day["t_mean"].iloc[ind-1:ind+1]))
    t3.append(np.mean(df_atm_day["t_mean"].iloc[ind-2:ind+1]))
    t4.append(np.mean(df_atm_day["t_mean"].iloc[ind-3:ind+1]))
    t5.append(np.mean(df_atm_day["t_mean"].iloc[ind-4:ind+1]))

    # calculate the average temperature over multiple days
    w2.append(np.mean(df_atm_day["wspeed_mean"].iloc[ind-1:ind+1]))
    w3.append(np.mean(df_atm_day["wspeed_mean"].iloc[ind-2:ind+1]))
    w4.append(np.mean(df_atm_day["wspeed_mean"].iloc[ind-3:ind+1]))
    w5.append(np.mean(df_atm_day["wspeed_mean"].iloc[ind-4:ind+1]))

    # calculate the average maximum temperature over multiple days
    tmax2.append(np.mean(df_atm_day["t_max"].iloc[ind-1:ind+1]))
    tmax3.append(np.mean(df_atm_day["t_max"].iloc[ind-2:ind+1]))
    tmax4.append(np.mean(df_atm_day["t_max"].iloc[ind-3:ind+1]))
    tmax5.append(np.mean(df_atm_day["t_max"].iloc[ind-4:ind+1]))

    # calculate the average maximum wind speed over multiple days
    wmax2.append(np.mean(df_atm_day["wspeed_max"].iloc[ind-1:ind+1]))
    wmax3.append(np.mean(df_atm_day["wspeed_max"].iloc[ind-2:ind+1]))
    wmax4.append(np.mean(df_atm_day["wspeed_max"].iloc[ind-3:ind+1]))
    wmax5.append(np.mean(df_atm_day["wspeed_max"].iloc[ind-4:ind+1]))

    # calculate the temperature amplitude the day before and up to three days before the event
    dtemp1.append(df_atm_day["t_range"].iloc[ind-1])
    dtemp2.append(df_atm_day["t_range"].iloc[ind-2])
    dtemp3.append(df_atm_day["t_range"].iloc[ind-3])

    # calculate the temperature amplitude between 1-3 days before and the day of the event
    dtempd1.append(np.max([df_atm_day["t_max"].iloc[ind-1] - df_atm_day["t_min"].iloc[ind],
                           df_atm_day["t_max"].iloc[ind] - df_atm_day["t_min"].iloc[ind-1]]))
    dtempd2.append(np.max([df_atm_day["t_max"].iloc[ind-2] - df_atm_day["t_min"].iloc[ind],
                           df_atm_day["t_max"].iloc[ind] - df_atm_day["t_min"].iloc[ind-2]]))
    dtempd3.append(np.max([df_atm_day["t_max"].iloc[ind-3] - df_atm_day["t_min"].iloc[ind],
                           df_atm_day["t_max"].iloc[ind] - df_atm_day["t_min"].iloc[ind-3]]))

    # in lieu of having the "thaw periods" (whatever they exactly mean by that in the paper), calculate the sum of the
    # positive degree-days for a seven-day period before and including an event-day (i.e., six days prior and the day
    # itself)
    sev_temp = np.array(df_atm_day["t_mean"].iloc[ind-6:ind+1]) - 273.15
    pdd.append(np.sum(sev_temp[sev_temp > 0]))

    # wind drift (see Hendriks et al. 2014)
    avg_wspeed2.append(np.mean(df_atm_day["wspeed_mean"].iloc[ind-1:ind+1]))
    avg_wspeed3.append(np.mean(df_atm_day["wspeed_mean"].iloc[ind-2:ind+1]))

    count += 1
    print_progress_bar(count, l, prefix='Progress:', suffix='Complete', length=50)
# end for da

print(f"\nWent through {count} iterations.\n")

wdrift_2 = np.array(s2) * np.array(avg_wspeed2)
wdrift_3 = np.array(s3) * np.array(avg_wspeed3)
wdrift3_2 = np.array(s2) * np.array(avg_wspeed2)**3
wdrift3_3 = np.array(s3) * np.array(avg_wspeed3)**3


#%% add the accrued quantities to the dataframe
df_atm_risk["r2"] = np.array(r2)
df_atm_risk["r3"] = np.array(r3)
df_atm_risk["r4"] = np.array(r4)
df_atm_risk["r5"] = np.array(r5)
df_atm_risk["r6"] = np.array(r6)
df_atm_risk["r7"] = np.array(r7)

df_atm_risk["s2"] = np.array(s2)
df_atm_risk["s3"] = np.array(s3)
df_atm_risk["s4"] = np.array(s4)
df_atm_risk["s5"] = np.array(s5)
df_atm_risk["s6"] = np.array(s6)
df_atm_risk["s7"] = np.array(s7)

df_atm_risk["t2"] = np.array(t2)
df_atm_risk["t3"] = np.array(t3)
df_atm_risk["t4"] = np.array(t4)
df_atm_risk["t5"] = np.array(t5)

df_atm_risk["tmax2"] = np.array(tmax2)
df_atm_risk["tmax3"] = np.array(tmax3)
df_atm_risk["tmax4"] = np.array(tmax4)
df_atm_risk["tmax5"] = np.array(tmax5)

df_atm_risk["w2"] = np.array(w2)
df_atm_risk["w3"] = np.array(w3)
df_atm_risk["w4"] = np.array(w4)
df_atm_risk["w5"] = np.array(w5)

df_atm_risk["wmax2"] = np.array(wmax2)
df_atm_risk["wmax3"] = np.array(wmax3)
df_atm_risk["wmax4"] = np.array(wmax4)
df_atm_risk["wmax5"] = np.array(wmax5)

df_atm_risk["dtemp1"] = np.array(dtemp1)
df_atm_risk["dtemp2"] = np.array(dtemp2)
df_atm_risk["dtemp3"] = np.array(dtemp3)

df_atm_risk["dtempd1"] = np.array(dtempd1)
df_atm_risk["dtempd2"] = np.array(dtempd2)
df_atm_risk["dtempd3"] = np.array(dtempd3)

df_atm_risk["pdd"] = np.array(pdd)

# df_atm_risk["wdrift"] = np.array(df_atm_risk["wspeed_mean"]) * np.array(df_atm_risk["total_prec_sum"])
# df_atm_risk["wdrift3"] = np.array(df_atm_risk["wspeed_mean"]**3) * np.array(df_atm_risk["total_prec_sum"])

df_atm_risk["wdrift"] = np.array(df_atm_risk["wspeed_mean"]) * np.array(df_atm_risk["s1"])
df_atm_risk["wdrift3"] = np.array(df_atm_risk["wspeed_mean"]**3) * np.array(df_atm_risk["s1"])

df_atm_risk["wdrift_2"] = np.array(wdrift_2)
df_atm_risk["wdrift3_2"] = np.array(wdrift3_2)
df_atm_risk["wdrift_3"] = np.array(wdrift_3)
df_atm_risk["wdrift3_3"] = np.array(wdrift3_3)


#%% store the data as csv
out_path = f"/PATH_TO_AVALANCHE_PREDICTORS/Between{h_low}_and_{h_hi}m/"
# --> will be generated

os.makedirs(out_path, exist_ok=True)
df_atm_risk.to_csv(out_path + region + f"_Predictors_MultiCellMean_Between{h_low}_and_{h_hi}m{expos_add}.csv")


#%% store the lat and lon values of the individual grids over which the data was averaged
os.makedirs(out_path + "Lats_and_Lons/", exist_ok=True)
lon_lat.to_csv(out_path + "Lats_and_Lons/" + region +
               f"_MultiCell_Lat_and_Lon_Between{h_low}_and_{h_hi}m{expos_add}.csv")
