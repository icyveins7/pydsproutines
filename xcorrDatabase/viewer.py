from ._core import *

import dearpygui.dearpygui as dpg
import os
import time

#%% Boilerplate
dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()


#%% Database file opener window on start-up
xdb = dict() # To store database objects, path->database object
# xdb_windows = dict() # To store windows, path->window object

def open_db():
    dbpath = dpg.get_value("dbpathinput")
    if dbpath in xdb:
        dpg.set_value("opendb_result", "Database already opened.")

    elif os.path.exists(dbpath):
        xdb[dbpath] = XcorrDB(dbpath)
        dpg.set_value("opendb_result", "Successfully opened database.")
        print(xdb)
        # Add a window
        dpg.add_window(
            label=dbpath, indent=1,
            width=700, tag="window_%s" % (dbpath),
            on_close=clear_db_window, user_data=dbpath
        )
    
    else:
        dpg.set_value("opendb_result", "Database not found. Check your path.")


def set_db_path(sender, app_data):
    dpg.set_value("dbpathinput", app_data['file_path_name'])

with dpg.file_dialog(
    show=False, 
    callback=set_db_path, 
    tag="file_dialog_id",
    width=700 ,height=400
):
    dpg.add_file_extension(".db")
    dpg.add_file_extension(".*")

with dpg.window(label="Open an XcorrDB", no_close=True, width=700):
    with dpg.group(horizontal=True):
        dpg.add_input_text(label="DB path", default_value="xcorrs.db", callback=open_db, tag="dbpathinput")
        dpg.add_button(label="Browse", callback=lambda: dpg.show_item("file_dialog_id"))

    dpg.add_button(label="Open", callback=open_db)
    dpg.add_text(tag="opendb_result")

    for dbpath in xdb:
        dpg.add_text("Opened: %s" % (dbpath))

#%%
def clear_db_window(sender, app_data, user_data):
    print(user_data)
    xdb.pop(user_data)
    dpg.delete_item(user_data)

    
#%% Boilerplate
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()