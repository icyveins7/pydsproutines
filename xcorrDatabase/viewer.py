from ._core import *

import dearpygui.dearpygui as dpg

#%% Boilerplate
dpg.create_context()
dpg.create_viewport()
dpg.setup_dearpygui()


#%% Main window on start-up
xdb = [] # To store database objects (may open more than 1)

def open_db():
    xdb.append(XcorrDB(dpg.get_value("dbpathinput")))
    print(xdb)

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
    
#%% Boilerplate
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()