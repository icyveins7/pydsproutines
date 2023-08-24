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
        dpg.add_input_text(label="DB path", default_value="xcorrs.db", tag="dbpathinput", callback=open_db, on_enter=True)
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
                    

#%%
def handleColumnToggle(sender, app_data, user_data):
    # Get the target tag string (tablename)
    targetTag = user_data['targetTag']
    # Get the column name string
    column = user_data['column']
    # Get the current state of the checkbox
    state = dpg.get_value(sender)
    # Get the table widget?
    table = dpg.get_value(targetTag + "_table") # This is the format of the table tag

    print("Editing table %s, column %s" % (targetTag, column))
    print(state)
    print(table) # Not useful


#%% Holder for toggling the current textviewer type
textviewerType = dict() # tag->bool (isHex)

#%%
def clearDataWindow(sender, app_data, user_data):
    # Clear things with custom tags
    dpg.delete_item(user_data['textviewertag'])
    textviewerType.pop(user_data['textviewertag'])


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
        width=700, height=400,
        # parent="window_%s" % (dbpath) # This doesn't work?
        on_close=clearDataWindow,
        user_data={
            'textviewertag': "textviewer%s,%s" % (dbpath,tablename)
        }
    ):
        # Get the table type
        xctype = table.xctype

        # Define the format based on type
        if xctype == 0:
            fmt = XcorrDB._xcorr_type0results_fmt()
        elif xctype == 1:
            fmt = XcorrDB._xcorr_type1results_fmt()
        elif xctype == 2:
            raise NotImplementedError("2D not implemented yet")        
        else:
            raise ValueError("Undefined type %d" % xctype)

        
        # We use table for horizontal layout
        tbl_cols = list()

        with dpg.table(header_row=False, borders_innerV=True):
            # Column for the actual table
            dpg.add_table_column()
            # Column for the blob display
            dpg.add_table_column(width_fixed=True, init_width_or_weight=200)
            with dpg.table_row():
                # Middle layout: 
                with dpg.table_cell():
                    # Table for the data
                    with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp, hideable=True): # , tag="%s_table" % tablename):
                        # Different xctypes have different displays
                        if xctype == 0:
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
                            ignoredColumns = ["qf2", "freqIdx", "rfdIdx", "View"] # The view column holds the button
                            # Add columns as-is, but ignore the result blobs
                            for col in fmt['cols']:
                                if col[0] not in ignoredColumns:
                                    tbl_cols.append(dpg.add_table_column(label=col[0]))
                            
                            # Add plot button column
                            dpg.add_table_column(label="View", width_fixed=True, init_width_or_weight=40.0) # We want the button to show up fully
                            
                            # Add the rows
                            hideRfdCols = True
                            for row in results:
                                with dpg.table_row():
                                    for idx, col in enumerate(row):
                                        if fmt['cols'][idx][0] not in ignoredColumns:
                                            with dpg.table_cell(): # In case we need more things per cell
                                                if isinstance(col, bytes):
                                                    dpg.add_button(
                                                        label="BLOB",
                                                        callback=renderBlobText,
                                                        user_data={
                                                            "blob": col,
                                                            "textviewertag": "textviewer%s,%s" % (dbpath,tablename)
                                                        }
                                                    )
                                                    
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
                                # Check if all of rfd is None
                                if row['rfd_scan_start'] is not None or row['rfd_scan_numsteps'] is not None or row['rfd_scan_step'] is not None:
                                    hideRfdCols = False

                                if hideRfdCols:
                                    # TODO: have modal to tell user that we hid these
                                    dpg.disable_item(tbl_cols[-4])
                                    dpg.show_item(tbl_cols[-4]) # Need this as well?
                                    dpg.disable_item(tbl_cols[-3])
                                    dpg.show_item(tbl_cols[-3])
                                    dpg.disable_item(tbl_cols[-2])
                                    dpg.show_item(tbl_cols[-2])
                                
                # Right layout: 
                with dpg.table_cell() as righttblpanel:
                    # Text viewer
                    textviewerTag = "textviewer%s,%s" % (dbpath,tablename)
                    dpg.add_input_text(
                        multiline=True, 
                        default_value="Nothing to show here.", 
                        width=200,
                        height=300, 
                        #callback=_log, s
                        #tab_input=True,
                        readonly=True,
                        parent=righttblpanel,
                        tag=textviewerTag)

                    with dpg.group(horizontal=True):
                        dpg.add_button(
                            label="Toggle Hex/uint8s", 
                            callback=toggleTextViewerOutput,
                            user_data={
                                'textviewertag': textviewerTag
                            })

#%% Callback for blob renderer
def renderBlobText(sender, app_data, user_data):
    blob = user_data['blob']
    x = np.frombuffer(blob, np.uint8)
    # Is this the first time we render?
    if user_data['textviewertag'] not in textviewerType:
        textviewerType[user_data['textviewertag']] = True # Default isHex = True
    
    if textviewerType[user_data['textviewertag']]: # if True, use hex
        s = " ".join(["%02X" % i for i in x])
    else:
        s = " ".join(["%3d" % i for i in x])

    dpg.set_value(user_data['textviewertag'], s)


#%% Callback to toggle text viewer output
def toggleTextViewerOutput(sender, app_data, user_data):
    current = dpg.get_value(user_data['textviewertag'])
    if textviewerType[user_data['textviewertag']]: # Then it is hex
        u8rep = np.frombuffer(bytes.fromhex(current), np.uint8)
        s = " ".join(["%3d" % i for i in u8rep])

    else:
        hexrep = np.array([int(i) for i in current.split()], np.uint8)
        s = " ".join(["%02X" % i for i in hexrep])

    # Set the new representation
    dpg.set_value(user_data['textviewertag'], s)
    textviewerType[user_data['textviewertag']] = not textviewerType[user_data['textviewertag']] # Don't forget to invert the bool
        

#%% Callback for 1D/2D Plotter
def plotDataWindow(sender, app_data, user_data):
    row = user_data['row']
    table = user_data['table']

    if table.xctype == XcorrDB.TYPE_1D:
        tdrange, qf2, freqinds = table.regenerate1Dresults(row)
        mi = np.argmax(qf2)

        # Create the plot window
        with dpg.window(label="1D QF2 Result"):
            with dpg.subplots(2, 1, link_all_x=True, height=600, width=600):
                with dpg.plot(label="QF2 vs TD"):
                    tdqf2axis = dpg.add_plot_axis(dpg.mvXAxis, label="TD")
                    qf2axis = dpg.add_plot_axis(dpg.mvYAxis, label="QF2")
                    dpg.add_line_series(tdrange, qf2, parent=qf2axis) # Dynamically point to the parent
                    dpg.fit_axis_data(qf2axis)
                    dpg.fit_axis_data(tdqf2axis)

                with dpg.plot(label="Frequency vs TD"):
                    tdfreqaxis = dpg.add_plot_axis(dpg.mvXAxis, label="TD")
                    freqaxis = dpg.add_plot_axis(dpg.mvYAxis, label="Frequency")
                    dpg.add_line_series(tdrange, freqinds.astype(np.float64), parent=freqaxis)
                    dpg.fit_axis_data(qf2axis)
                    dpg.fit_axis_data(tdfreqaxis)

            
            dpg.add_text("Peak QF2: %.6g" % (qf2[mi]))
            dpg.add_text("Freq. Index at Peak: %d" % (freqinds[mi]))
            dpg.add_text("TDOA at Peak: %.6g" % (tdrange[mi]))
    

#%% Boilerplate
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()