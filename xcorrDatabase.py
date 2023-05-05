"""
Here we will describe the schema used to store several generic configurations of
xcorr results.

Hopefully, this will be useful for many cases and can be widely distributed, so that a common way of 
storing and retrieving xcorr results can be finalised.

This is largely in collaboration with https://github.com/car-engine/gl-sql, where the schemas/table formats
should be roughly the same or identical, but here is Pythonic rather than in MATLAB.

Requirements:
1) https://github.com/icyveins7/sew
This is my sqlite3 wrapper. I will be using this as the base, rather than the default sqlite3 module.
"""

import sew

from copy import deepcopy


#%% Define the class
class XcorrDB(sew.Database):
    ### Define the table schemas
    # Metadata Table
    xcorr_metadata_tblname = "xcorr_metadata"
    xcorr_metadata_fmt = {
        'cols': [
            ["data_tblname", "TEXT"], # The table that contains the xcorr results, adhering to sew's MetaTable
            ["fc", "REAL"], # Centre frequency of arrays used in the xcorr (Needed for Doppler compensation)
            ["fs", "INTEGER"], # Sample rate of arrays used in the xcorr
            ["s1", "TEXT"], # Arbitrary name to identify source 1
            ["s2", "TEXT"], # Arbitrary name to identify source 2
            ["xctype", "INTEGER"], # 0: peak values only, 1: 1-D results, 2: 2-D results
            ["desc", "BLOB"], # Arbitrary extra description parameters for storage
        ],
        'conds': [
            "UNIQUE(data_tblname)"
        ]
    }

    # Results Table (Base, all the formats will contain this)
    xcorr_results_fmt = {
        'cols': [
            ["time_sec", "INTEGER"], # GPS time of cutout
            ["tidx", "INTEGER"], # Index of cutout from source 1
            ["cutoutlen", "INTEGER"], # Length of cutout array
            ["td_scan_start", "REAL"], # Offset from tidx to start scanning at, in seconds
            ["td_scan_end", "REAL"], # Offset from tidx to end scanning at, in seconds
            ["td_scan_step", "REAL"], # Scan step size, in seconds (set to 1 sample duration if no subsample scans)
            ["fd_scan_start", "REAL"], # Similar ranges for FDOA
            ["fd_scan_end", "REAL"], # Perhaps set to all 0s if FFT resolution is used?
            ["fd_scan_step", "REAL"], # Otherwise this is assumed to be using a CZT?
            ["rfd_scan_start", "REAL"], # Similar ranges for RFDOA
            ["rfd_scan_end", "REAL"],
            ["rfd_scan_step", "REAL"]
        ],
        'conds': []
    }

    @staticmethod
    def _xcorr_type0results_fmt(self):
        """
        Returns type 0 results table format;
        this table contains scalar peak values only.
        """
        fmt = deepcopy(self.xcorr_results_fmt) # We don't want to modify the static formats
        fmt['cols'].extend([
            ["qf2", "REAL"],
            ["td", "REAL"],
            ["td_sigma", "REAL"],
            ["fd", "REAL"],
            ["fd_sigma", "REAL"],
            ["rfd", "REAL"],
            ["rfd_sigma", "REAL"]
        ])
        return fmt
    
    @staticmethod
    def _xcorr_type1results_fmt(self):
        """
        Returns type 1 results table format;
        this table contains 1-D blob results, usually where the
        max value of the CAF has been selected for every time shift.
        """
        fmt = deepcopy(self.xcorr_results_fmt) # We don't want to modify the static formats
        fmt['cols'].extend([
            ["qf2", "BLOB"],
            ["freqIdx", "BLOB"],
            ["rfdIdx", "BLOB"]
        ])
        return fmt
    
    @staticmethod
    def _xcorr_type2results_fmt(self):
        """
        Returns type 2 results table format;
        this table contains 2-D blob results, like a full CAF.
        """
        fmt = deepcopy(self.xcorr_results_fmt) # We don't want to modify the static formats
        fmt['cols'].extend([
            ["caf", "BLOB"]
        ])
        return fmt


    ########## Constructor
    def __init__(self, *args, **kwargs):
        """
        Forwards all constructor arguments to sew.Database.__init__().
        """
        super().__init__(*args, **kwargs)

        # Always create the xcorr metadata table
        self._createMetaXcorrsTable()

    def _createMetaXcorrsTable(self):
        """
        This does not need to be called by the user usually.
        """
        self.createMetaTable(
            self.xcorr_metadata_fmt,
            self.xcorr_metadata_tblname
        )
    
    def createXcorrResultsTable(
        self,
        results_tblfmt: dict,
        results_tblname: str,
        metadata: list,
        **kwargs
    ):
        """
        Creates a new xcorr results table.
        This adheres to the metadata/data table rules in the 'sew' package.

        Parameters
        ----------
        results_tblfmt : dict
            The format of the xcorr results table.
            This can usually be one of the following:
            XcorrDB._xcorr_type0results_fmt,
            XcorrDB._xcorr_type1results_fmt,
            XcorrDB._xcorr_type2results_fmt
        results_tblname : str
            The name of the xcorr results table.
        metadata : list
            The metadata for the xcorr results table. This will be inserted into
            the metadata table.
        """
        
        self.createDataTable(
            results_tblfmt,
            results_tblname,
            metadata,
            self.xcorr_metadata_tblname,
            **kwargs
        )

