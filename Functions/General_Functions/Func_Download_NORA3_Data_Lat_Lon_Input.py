#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use the functions provided in the metocean-api to download NORA3 subsets. The "novelty" here is that the function will
allow to generate >>spatial<< subsets and not only temporal subsets.
"""

#%% imports --> import my adjusted ts module
import os
import sys
from nco import Nco
from metocean_api import ts_KUE as ts


#%% function
def download_nora3(lon1, lat1, lon2, lat2, product='NORA3_atm_sub', start_date='1985-01-01', end_date='1985-01-31',
                   var_l=["air_temperature_2m"],
                   outpath="/media/kei070/One Touch/IMPETUS/NORA3/",
                   outname="NORA3_sub.nc",
                   gen_dir=True):

    """
    Function for downloading NORA3 data subsets employing some of the metocean-api functions. The novelty of the
    function is that in addition to temporal also >>spatial<< subsets can be created. That is, here we download gridded
    data instead of time-series data for individual gridcells.

    Parameters:
        lon1:    Float. Longitude of the south-west corner of the spatial subset. Units: degrees east.
        lat1:    Float. Latitude of the south-west corner of the spatial subset. Units: degrees north.
        lon2:    Float. Longitude of the north-east corner of the spatial subset. Units: degrees east.
        lat2:    Float. Latitude of the north-east corner of the spatial subset. Units: degrees north.
        product: String. The NORA3 product name. Defaults to NORA3_atm_sub.
        start_date: String. Start data of temporal subset. Format: YYYY-MM-DD. Defaults to 1985-01-01.
        end_date: String. End data of temporal subset. Format: YYYY-MM-DD. Defaults to 1985-01-31.
        var_l: List of strings. A list of the names of variables to be downloaded. Defaults to ["air_temperature_2m"].
        outpath: String. Path to the directory where the downloaded data will be stored.
        outname: String. Name of the netcdf file containing the downloaded data. Defaults to NORA3_sub.nc.
        gen_dir: Boolean. If True (default), the directory specified in the parameter outpath will be created.
    """

    # generate the output director if requested
    if gen_dir:
        os.makedirs(outpath, exist_ok=True)
    # ednd if

    # generate the date list and get the URL info of the file
    date_list = ts.get_date_list(product=product, start_date=start_date, end_date=end_date)
    x_coor_str, y_coor_str, infile = ts.get_url_info(product, date_list[0])

    # get the nearest coordinates on the NORA3 grid
    x_coor, y_coor, lon_near, lat_near = ts.get_near_coord(infile=infile, lon=lon1, lat=lat1, product=product)
    x_coor2, y_coor2, lon_near2, lat_near2 = ts.get_near_coord(infile=infile, lon=lon2, lat=lat2, product=product)

    # construct the NCO command
    opt = ['-O -v ' + ",".join(var_l) + ",longitude,latitude" +
           f' -d x,{x_coor.values[0]},{x_coor2.values[0]} -d y,{y_coor.values[0]},{y_coor2.values[0]}']

    # print the NCO command
    print("\nThe NCO command is:")
    print(opt[0])
    print("\n")

    # inform the user if x2 < x1 and y2 < y1
    if x_coor2.values[0] < x_coor.values[0]:
        print("\nNote that x2 < x1\n")
    if y_coor2.values[0] < y_coor.values[0]:
        print("\nNote that y2 < y1\n")
    # end if

    # ask the user if - after seeing the NCO command - the execution should be stopped
    stop_exe = input("\nStop the execution? (Y/no)\n")
    if stop_exe == "":
        stop_exe = "y"
    if (stop_exe.lower() == "y") | (stop_exe.lower() == "yes"):
        sys.exit("Stopping execution.")
    # end if

    # execute the NCO commend and subsequently the download
    nco = Nco()
    nco.ncks(input=infile, output=outpath+outname, options=opt)
# end def
