from ._core import *

import dearpygui.dearpygui as dpg
import os
import time
import numpy as np

#%% Boilerplate
dpg.create_context()
dpg.create_viewport(title="XcorrDatabase Viewer")
dpg.setup_dearpygui()


#%% Internal holder for databases
xdb = dict() # To store database objects, path->database object
# xdb_windows = dict() # To store windows, path->window object

#%% Callback to open database and associated window
def open_db():
    dbpath = dpg.get_value("dbpathinput")
    if dbpath in xdb:
        dpg.set_value("opendb_result", "Database already opened.")

    elif os.path.exists(dbpath):
        xdb[dbpath] = XcorrDB(dbpath)
        dpg.set_value("opendb_result", "Successfully opened database.")
        print(xdb)
        # Setup the db-specific window
        setupDbWindow(dbpath)

        # Add text
        dpg.add_text("Opened: %s" % (dbpath), parent="opener_window", tag="opened_text_%s" % (dbpath))
    
    else:
        dpg.set_value("opendb_result", "Database not found. Check your path.")

#%% Callback to fill text input from file selector
def set_db_path(sender, app_data):
    dpg.set_value("dbpathinput", app_data['file_path_name'])

#%% Setup for file selector
with dpg.file_dialog(
    show=False, 
    callback=set_db_path, 
    tag="file_dialog_id",
    width=700 ,height=400
):
    dpg.add_file_extension(".db")
    dpg.add_file_extension(".*")

#%% Starting window, to open databases
with dpg.window(label="Open an XcorrDB", no_close=True, width=500, height=200, tag="opener_window"):
    with dpg.group(horizontal=True):
        dpg.add_input_text(label="DB path", default_value="xcorrs.db", tag="dbpathinput")
        dpg.add_button(label="Browse", callback=lambda: dpg.show_item("file_dialog_id"))

    dpg.add_button(label="Open", callback=open_db)
    dpg.add_text(tag="opendb_result")

#%% Callback when database window is closed
def clear_db_window(sender, app_data, user_data):
    print("closing db window %s" % user_data)
    xdb.pop(user_data)
    print(xdb)
    # Delete the window
    dpg.delete_item("window_%s" % user_data)
    # Delete the text in opener
    dpg.delete_item("opened_text_%s" % user_data)

#%% Function to setup a database metadata window
def setupDbWindow(dbpath):
    # Open the metadata and extract show available tables
    db = xdb[dbpath]
    db.reloadTables()
    db['xcorr_metadata'].select("*")
    metadata = db.fetchall()

    # Create the window
    parentWindowTag = "window_%s" % (dbpath)
    with dpg.window(
        label=dbpath, pos=(200 + len(xdb)*10, 200),
        width=500, height=400, 
        tag=parentWindowTag, # if you specify tag, you must delete_item yourself i think
        on_close=clear_db_window, user_data=dbpath
    ):
        # Add the table
        with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
            for col in XcorrDB.xcorr_metadata_fmt['cols']:
                dpg.add_table_column(label=col[0])
            dpg.add_table_column(label="View", width_fixed=True, init_width_or_weight=40.0) # One more column to store the button for opening

            # Add the rows extracted
            for row in metadata:
                with dpg.table_row():
                    for i in row:
                        with dpg.table_cell(): # In case we need more things per cell
                            dpg.add_text(str(i))
                    # Add the button
                    dpg.add_button(label="Load", callback=setupDataWindow, user_data={
                        "dbpath": dbpath,
                        "table": row['data_tblname'],
                    })
                    
#%% Function to setup a database data window
def setupDataWindow(sender, app_data, user_data):
    dbpath = user_data['dbpath']
    db = xdb[dbpath]
    tablename = user_data['table']
    table = db[tablename]
    xctype = table.xctype
    xctypestrdict = {
        XcorrDB.TYPE_PEAKVALUES: "Scalar Peak Value",
        XcorrDB.TYPE_1D: "1D",
        XcorrDB.TYPE_2D: "2D"
    }

    table.select("*")
    results = db.fetchall()

    # Create the window
    with dpg.window(
        label="%s -> %s (%s Results)" % (dbpath, tablename, xctypestrdict[xctype]),
        pos=(300, 300),
        width=500, height=400,
        # parent="window_%s" % (dbpath) # This doesn't work?
    ):
        # Get the table type
        xctype = table.xctype

        with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
            # Different xctypes have different displays
            if xctype == 0:
                fmt = XcorrDB._xcorr_type0results_fmt()
                # Add columns as-is
                for col in fmt['cols']:
                    dpg.add_table_column(label=col[0])
                
                # Add the rows
                for row in results:
                    with dpg.table_row():
                        for i in row:
                            with dpg.table_cell(): # In case we need more things per cell
                                if isinstance(i, bytes):
                                    dpg.add_text("BLOB")
                                else:
                                    dpg.add_text(str(i))

            elif xctype == 1:
                fmt = XcorrDB._xcorr_type1results_fmt()
                ignoredColumns = ["qf2", "freqIdx", "rfdIdx", "View"] # The view column holds the button
                # Add columns as-is, but ignore the result blobs
                for col in fmt['cols']:
                    if col[0] not in ignoredColumns:
                        dpg.add_table_column(label=col[0])
                
                # Add plot button column
                dpg.add_table_column(label="View", width_fixed=True, init_width_or_weight=40.0) # We want the button to show up fully
                
                # Add the rows
                for row in results:
                    with dpg.table_row():
                        for idx, col in enumerate(row):
                            if fmt['cols'][idx][0] not in ignoredColumns:
                                with dpg.table_cell(): # In case we need more things per cell
                                    if isinstance(col, bytes):
                                        dpg.add_text("BLOB")
                                    else:
                                        dpg.add_text(str(col))
                        with dpg.table_cell():
                            dpg.add_button(
                                label="Plot", 
                                callback=plotDataWindow, 
                                user_data={
                                    "row": row, # Send the row
                                    "table": table # And the table object
                                }
                            )

            elif xctype == 2:
                raise NotImplementedError("TODO: handle type2")
                # fmt = XcorrDB._xcorr_type2results_fmt()
            else:
                raise ValueError("Undefined type %d" % xctype)

#%% Callback for 1D/2D Plotter
def plotDataWindow(sender, app_data, user_data):
    row = user_data['row']
    table = user_data['table']

    if table.xctype == XcorrDB.TYPE_1D:
        tdrange, qf2, freqinds = table.regenerate1Dresults(row)

        # Create the plot window
        with dpg.window(label="1D QF2 Result"):
            with dpg.plot(label="QF2 vs TD", height=600, width=600):
                dpg.add_plot_axis(dpg.mvXAxis, label="TD")
                yaxis = dpg.add_plot_axis(dpg.mvYAxis, label="QF2")
                dpg.add_line_series(tdrange, qf2, parent=yaxis) # Dynamically point to the parent
    

#%% Boilerplate
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()