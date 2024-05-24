"""
This is the TimeSeries-module that is responsible for extracting TimeSeries-object.

The Extraction-object and the abstract classes used by the other sub-
modules are defined in ts_mod.py, and this should not be modified.

In addition it contains the following sub-modules, which can be expanded by
the user:

read: Contains classes that are used to read grid data from different sources.

Copyright 2023, Konstantinos Christakos, MET Norway
"""

from .ts_mod import TimeSeries
from .read_metno import *
from .read_ec import *
from .aux_funcs import get_date_list, get_url_info, get_near_coord, create_dataframe, check_datafile_exists, read_commented_lines
