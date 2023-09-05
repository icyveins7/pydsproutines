from .._core import *
from .editor import ConfigEditor

import dearpygui.dearpygui as dpg

#%% Boilerplate
dpg.create_context()
dpg.create_viewport(title="PyDSP Config Editor")
dpg.setup_dearpygui()

#%% Instantiate the editor and run
editor = ConfigEditor()
editor.run()

#%% Boilerplate
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()