import dearpygui.dearpygui as dpg
import os

from .._core import *
from .helpers import centreModal
from .editConfig import EditConfigWindow

#%%
class OpenConfigDialog:
    def __init__(self):
        # Key: path, value: tuple(add_text widget, EditConfigWindow)
        self.cfgDict = dict()

    def run(self):
        # Show a window that lets you open a config file
        self.window = dpg.add_window(
            label="Open a config file", 
            width=700, height=150,
            no_close=True)
        
        with dpg.group(horizontal=True, parent=self.window):
            self.cfgPathInput = dpg.add_input_text(
                label="Config filepath",
                callback=self._open_config,
                on_enter=True
            )
            dpg.focus_item(self.cfgPathInput)
            dpg.add_button(
                label="Browse",
                callback=self._config_file_selector
            )

        dpg.add_button(
            label="Open config",
            callback=self._open_config,
            parent=self.window
        )
        dpg.add_separator(parent=self.window)

    def _config_file_selector(self, sender, app_data, user_data):
        with dpg.file_dialog(
            callback=lambda dialog, dialog_data: [dpg.set_value(
                self.cfgPathInput, # This is the tag for the path input passed in
                item # This is the key for the path selected
            ) for key, item in dialog_data['selections'].items()],
            # Although we use a list comprehension, we assume only 1 item, so it reads the last one
            height=400, width=700,
        ):
            dpg.add_file_extension(".ini")
            dpg.add_file_extension(".*")

    def _open_config(self, sender, app_data, user_data):
        cfgpath = dpg.get_value(self.cfgPathInput)
        if os.path.exists(cfgpath):
            # Then open it
            self._add_editConfig(cfgpath, isNew=False)

        else:
            # Ask whether we want to create the config
            with dpg.window(
                label="Path does not exist",
                width=350, # height=20, # this doesn't seem to do anything
                modal=True,
                show=True
            ) as popupModal:
                centreModal(popupModal)
                dpg.add_text(
                    "Create a new config at the specified path?",
                    wrap=dpg.get_item_width(popupModal) # this doesn't automatically resize
                )
                
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Yes",
                        callback=self._create_config,
                        user_data=[cfgpath, popupModal]
                    )
                    dpg.add_button(
                        label="No",
                        callback=self._create_config,
                        user_data=[None, popupModal]
                    )

        print(self.cfgDict)

    def _create_config(self, sender, app_data, user_data):
        cfgpath, popupModal = user_data
        # Create the new config
        if cfgpath is not None:
            self._add_editConfig(cfgpath, isNew=True)
        dpg.configure_item(popupModal, show=False)

    def _add_editConfig(self, cfgpath: str, isNew: bool=True):
        dspcfg = DSPConfig.new(cfgpath, allow_no_value=True) if isNew else DSPConfig(cfgpath)

        # Create the two widgets in this open config window
        cfgWidgetGroup = dpg.add_group(horizontal=True, parent=self.window)
        dpg.add_text(cfgpath, parent=cfgWidgetGroup)
        dpg.add_button(
            label="X",
            callback=self._del_editConfig,
            user_data=[cfgpath],
            parent=cfgWidgetGroup)

        # Store it
        self.cfgDict[cfgpath] = (
            cfgWidgetGroup,
            EditConfigWindow(dspcfg,cfgpath,self)
        )

    def _del_editConfig(self, sender, app_data, user_data):
        # To be called by the edit config window
        cfgpath, = user_data # Unpack the cfgpath which is the key
        # Pop it from the cfgDict
        cfgWidgets = self.cfgDict.pop(cfgpath)
        dpg.delete_item(cfgWidgets[0])
        dpg.delete_item(cfgWidgets[1].window)
        print("Cleaned up %s widgets" % (cfgpath))


