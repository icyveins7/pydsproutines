from ._core import *

import dearpygui.dearpygui as dpg
import os
import time

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
            dpg.add_table_column() # One more column to store the button for opening

            # Add the rows extracted
            for row in metadata:
                with dpg.table_row():
                    for i in row:
                        with dpg.table_cell(): # In case we need more things per cell
                            dpg.add_text(str(i))
                    # Add the button
                    dpg.add_button(label="Load", callback=setupDataWindow, user_data={
                        "dbpath": dbpath,
                        "table": row['data_tblname']
                    })
                    
#%% Function to setup a database data window
def setupDataWindow(sender, app_data, user_data):
    dbpath = user_data['dbpath']
    db = xdb[dbpath]
    tablename = user_data['table']
    table = db[tablename]

    table.select("*")
    results = db.fetchall()

    # Create the window
    with dpg.window(
        label="%s -> %s" % (dbpath, tablename),
        pos=(300, 300),
        width=500, height=400,
        # parent="window_%s" % (dbpath) # This doesn't work?
    ):
        # Get the table type
        xctype = table.xctype

        with dpg.table(header_row=True, resizable=True, policy=dpg.mvTable_SizingStretchProp):
            if xctype == 0:
                fmt = XcorrDB._xcorr_type0results_fmt()
            elif xctype == 1:
                fmt = XcorrDB._xcorr_type1results_fmt()
            elif xctype == 2:
                fmt = XcorrDB._xcorr_type2results_fmt()
            else:
                raise ValueError("Undefined type %d" % xctype)
            
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

    

    
#%% Boilerplate
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()