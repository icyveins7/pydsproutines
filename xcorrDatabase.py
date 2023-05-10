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
import numpy as np
import sqlite3
import matplotlib.pyplot as plt

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
            ["td_scan_start", "REAL"], # Offset from tidx to start scanning at, in seconds; generally this is at sample level
            ["td_scan_numsteps", "INTEGER"], # Number of scan steps; this is preferable to using an end value, as we prevent floating point ambiguities; this is also at sample level
            ["td_scan_step", "REAL"], # Scan step size, in seconds (if this is <1 sample duration, it is implicit that a subsample scan was done)
            ["fd_scan_start", "REAL"], # Similar ranges for FDOA
            ["fd_scan_numsteps", "INTEGER"], # Can be set to 0 if equal to cutoutlen i.e. natural FFT is used
            ["fd_scan_step", "REAL"], # Otherwise this is assumed to be using a CZT?
            ["rfd_scan_start", "REAL"], # Similar ranges for RFDOA
            ["rfd_scan_numsteps", "INTEGER"],
            ["rfd_scan_step", "REAL"],
            ["desc", "BLOB"] # Arbitrary extra description parameters for this particular xcorr
        ],
        'conds': [
            "UNIQUE(time_sec, tidx, cutoutlen, td_scan_start, td_scan_numsteps, td_scan_step, fd_scan_start, fd_scan_numsteps, fd_scan_step, rfd_scan_start, rfd_scan_numsteps, rfd_scan_step, desc)"
        ]
    }

    # Define constants for the types
    TYPE_PEAKVALUES = 0
    TYPE_1D = 1
    TYPE_2D = 2

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
            ["qf2", "BLOB"], # Recommended to be 64-bit floating
            ["freqIdx", "BLOB"], # Recommended to be 32-bit unsigned int
            ["rfdIdx", "BLOB"] # Recommended to be 32-bit unsigned int
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
        super().__init__(*args, **kwargs)

        # Always create the xcorr metadata table
        self._createMetaXcorrsTable()
        self.reloadTables()

    def _createMetaXcorrsTable(self):
        """
        This does not need to be called by the user usually.
        """
        self.createMetaTable(
            self.xcorr_metadata_fmt,
            self.xcorr_metadata_tblname,
            ifNotExists=True
        )

    def reloadTables(self):
        super().reloadTables()
        # We re-class our tables to match our internal formats, as a hotfix until sew is updated
        for tblname, table in self._tables.items():
            if isinstance(table, sew.MetaTableProxy):
                metatable = XcorrMetaTableProxy(
                    table._parent,
                    table._tbl,
                    table._fmt
                )
                self._tables[tblname] = metatable
            elif isinstance(table, sew.DataTableProxy):
                datatable = XcorrResultsTableProxy(
                    table._parent,
                    table._tbl,
                    table._fmt,
                    table._metadatatable
                )
                self._tables[tblname] = datatable

    
    def createXcorrResultsTable(
        self,
        results_tblname: str,
        fc: float,
        fs: int,
        s1: str,
        s2: str,
        xctype: int,
        desc: bytes = None,
        **kwargs
    ):
        """
        Creates a new xcorr results table.
        This adheres to the metadata/data table rules in the 'sew' package.

        Parameters
        ----------
        results_tblname : str
            The name of the xcorr results table.
        fc : float
            Centre frequency of arrays used in the xcorr.
        fs : int
            Sample rate of arrays used in the xcorr.
        s1 : str
            Name of source 1 (Generally where the cutout/preamble/template is from).
        s2 : str
            Name of source 2.
        xctype : int
            0: peak values only, 1: 1-D results, 2: 2-D results.
        desc : bytes, optional
            Any extra description parameters for storage.
        kwargs : dict
            Refer to sew.Database.createDataTable().
        """
        
        if xctype == 0:
            fmt = XcorrDB._xcorr_type0results_fmt(self)
        elif xctype == 1:
            fmt = XcorrDB._xcorr_type1results_fmt(self)
        elif xctype == 2:
            fmt = XcorrDB._xcorr_type2results_fmt(self)
        else:
            raise ValueError(f"xctype must be 0, 1, or 2, not {xctype}")

        self.createDataTable(
            fmt,
            results_tblname,
            [fc, fs, s1, s2, xctype, desc], # Format it into a list
            self.xcorr_metadata_tblname,
            **kwargs)

#%%
class XcorrMetaTableProxy(sew.MetaTableProxy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Should automatically refer to parent docstring

#%%
class XcorrResultsTableProxy(sew.DataTableProxy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) # Should automatically refer to parent docstring
        # Cache the metadata
        self._cacheMetadata = None

    # Define properties for the metadata
    @property
    def fc(self):
        self._cacheMetadata = self.getMetadata() if self._cacheMetadata is None else self._cacheMetadata
        return self._cacheMetadata["fc"]
    
    @property
    def fs(self):
        self._cacheMetadata = self.getMetadata() if self._cacheMetadata is None else self._cacheMetadata
        return self._cacheMetadata["fs"]
    
    @property
    def s1(self):
        self._cacheMetadata = self.getMetadata() if self._cacheMetadata is None else self._cacheMetadata
        return self._cacheMetadata["s1"]
    
    @property
    def s2(self):
        self._cacheMetadata = self.getMetadata() if self._cacheMetadata is None else self._cacheMetadata
        return self._cacheMetadata["s2"]
    
    @property
    def xctype(self):
        self._cacheMetadata = self.getMetadata() if self._cacheMetadata is None else self._cacheMetadata
        return self._cacheMetadata["xctype"]
    
    @property
    def desc(self):
        self._cacheMetadata = self.getMetadata() if self._cacheMetadata is None else self._cacheMetadata
        return self._cacheMetadata["desc"]

    ### Some helper functions
    def regenerateTDscanRange(self, td_scan_start: float, td_scan_numsteps: int, td_scan_step: float):
        return np.arange(td_scan_numsteps) * td_scan_step + td_scan_start

    def regenerate1Darray(self, qf2: bytes, dtype: type=np.float64):
        return np.frombuffer(qf2, dtype=dtype)

    def plot1Dresult(self, row: sqlite3.Row, qf2type: type=np.float64, freqindtype: type=np.uint32):
        tdrange = self.regenerateTDscanRange(
            row['td_scan_start'], 
            row['td_scan_numsteps'],
            row['td_scan_step'])
        qf2 = self.regenerate1Darray(row['qf2'], dtype=qf2type)
        freqinds = self.regenerate1Darray(row['freqIdx'], dtype=freqindtype)
        fig, ax = plt.subplots(2,1,num="%s, %d" % (self._tbl, row['time_sec']), sharex=True)
        ax[0].plot(tdrange, qf2)
        ax[1].plot(tdrange, freqinds)
        ax[1].set_xlabel("TDOA (s)")
        ax[0].set_ylabel("$QF^2$")
        ax[1].set_ylabel("Max Freq. Index")

        # Get the maximum and show it for convenience
        mi = np.argmax(qf2)
        tdest = tdrange[mi]
        qf2est = qf2[mi]
        freqIdxest = freqinds[mi]
        ax[0].plot(tdest, qf2est, 'rx')
        ax[1].plot(tdest, freqIdxest, 'rx')
        ax[0].set_title("$TD_{est} = %g, QF^2 = %g, f_i = %d$" % (tdest, qf2est, freqIdxest))

        return fig, ax


    
#%% Run some unit tests?
if __name__ == "__main__":
    import unittest

    class TestXcorrDB(unittest.TestCase):
        def setUp(self):
            self.db = XcorrDB(":memory:")

        # Check metadata table exists
        def test_checkMetaTable(self):
            self.assertTrue(
                self.db.xcorr_metadata_tblname in self.db.tables,
                "Metadata table was not found."
            )

        # Add a typical data table for type 0
        def test_addResultsTable(self):
            tblname = "results"

            self.db.createXcorrResultsTable(
                tblname,
                1e9,
                1000,
                "source1",
                "source2",
                0 # This is the type
            )
            self.db.reloadTables()
            self.assertTrue(
                tblname in self.db.tables,
                "Results table was not found."
            )

            # Retrieve the created table
            results = self.db[tblname]
            # Check that the metadata is correctly entered
            self.assertEqual(
                1e9, results.fc,
                "fc is not equal."
            )
            self.assertEqual(
                1000, results.fs,
                "fs is not equal."
            )
            self.assertEqual(
                "source1", results.s1,
                "s1 is not equal."
            )
            self.assertEqual(
                "source2", results.s2,
                "s2 is not equal."
            )
            self.assertEqual(
                0, results.xctype,
                "xctype is not equal."
            )
            self.assertEqual(
                None, results.desc,
                "desc is not equal."
            )

            # Insert some data
            # For type 0, format is
            # ["time_sec", "INTEGER"], # GPS time of cutout
            # ["tidx", "INTEGER"], # Index of cutout from source 1
            # ["cutoutlen", "INTEGER"], # Length of cutout array
            # ["td_scan_start", "REAL"], # Offset from tidx to start scanning at, in seconds
            # ["td_scan_numsteps", "INTEGER"], # Number of scan steps; this is preferable to using an end value, as we prevent floating point ambiguities
            # ["td_scan_step", "REAL"], # Scan step size, in seconds (set to 1 sample duration if no subsample scans)
            # ["fd_scan_start", "REAL"], # Similar ranges for FDOA
            # ["fd_scan_numsteps", "INTEGER"], # Can be set to 0 if equal to cutoutlen i.e. natural FFT is used
            # ["fd_scan_step", "REAL"], # Otherwise this is assumed to be using a CZT?
            # ["rfd_scan_start", "REAL"], # Similar ranges for RFDOA
            # ["rfd_scan_numsteps", "INTEGER"],
            # ["rfd_scan_step", "REAL"]
            # ["qf2", "REAL"],
            # ["td", "REAL"],
            # ["td_sigma", "REAL"],
            # ["fd", "REAL"],
            # ["fd_sigma", "REAL"],
            # ["rfd", "REAL"],
            # ["rfd_sigma", "REAL"]
            inserted = {
                "time_sec": 1234567890,
                "tidx": 123,
                "cutoutlen": 10000,
                "td_scan_start": -1e-3,
                "td_scan_numsteps": 10000,
                "td_scan_step": 1e-6,
                "qf2": 0.9,
                "td": 0,
                "td_sigma": 1e-7
            }
            results.insertOne(
                inserted,
                commitNow=True
            )
            # Extract the data and check
            results.select("*")
            r = self.db.fetchone()
            for k in inserted:
                self.assertEqual(
                    r[k], inserted[k]
                )

    unittest.main()
