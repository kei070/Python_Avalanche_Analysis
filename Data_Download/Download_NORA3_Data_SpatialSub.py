#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Employ the proprietary function download_nora3 to download a spatial subset of the NORA3 data.

This script only works with the metocean-api and my adjustment ts_KUE that has to be copied in there.

Link to metocean-api: https://github.com/MET-OM/metocean-api
"""


#%% imports
import os
import sys
import argparse


#%% set scripts directory and import self-written functions
script_direc = "/home/kei070/Documents/Python_Avalanche_Analysis/"
os.chdir(script_direc)
# needed for console
sys.path.append(script_direc)

from Functions.General_Functions.Func_Download_NORA3_Data import download_nora3


#%% load the parameters from the command line with argparse

# create the parser
parser = argparse.ArgumentParser(description="Download a spatio-temporal subset of the NORA3 data.")

# add arguments to the parser
parser.add_argument("x1", type=int, help="Use 390 for Troms")
parser.add_argument("y1", type=int, help="Use 800 for Troms")
parser.add_argument("x2", type=int, help="Use 530 for Troms")
parser.add_argument("y2", type=int, help="Use 1050 for Troms")
parser.add_argument("start_date", type=str, help="")
parser.add_argument("end_date", type=str, help="")

# parse the command-line arguments
args = parser.parse_args()


#%% download the data
download_nora3(x1=args.x1, y1=args.y1, x2=args.x2, y2=args.y2, product='NORA3_atm_sub',
               start_date=args.start_date, end_date=args.end_date,
               var_l=["air_temperature_2m", "precipitation_amount_hourly", "wind_speed", "wind_direction"],
               outpath="PATH_TO_NORA3/NORA3_NorthNorway_Sub/",
               outname_add="NORA3_NorthNorway_sub_",
               gen_dir=True)
