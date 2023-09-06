import dearpygui.dearpygui as dpg
import os

from .._core import *
from .helpers import centreModal

#%%
class OpenConfigDialog:
    def __init__(self):
        # List of tuples(DSPConfig, dpg.add_text)
        self.cfgs = list()

    def run(self):
        # Show a window that lets you open a config file
        with dpg.window(label="Open a config file", 
                        width=700, height=150,
                        no_close=True) as window:
            with dpg.group(horizontal=True):
                cfgPathInput = dpg.add_input_text(
                    label="Config filepath"
                )
                dpg.add_button(
                    label="Browse",
                    callback=self._config_file_selector,
                    user_data=cfgPathInput
                )
            dpg.add_button(
                label="Open config",
                callback=self._open_config,
                user_data=[cfgPathInput, window]
            )

        return 1

    def _config_file_selector(self, sender, app_data, user_data):
        """
        Expects user_data to be the input_text id/tag.
        """
        with dpg.file_dialog(
            callback=lambda dialog, dialog_data: [dpg.set_value(
                user_data, # This is the tag for the path input passed in
                item # This is the key for the path selected
            ) for key, item in dialog_data['selections'].items()],
            # Although we use a list comprehension, we assume only 1 item, so it reads the last one
            height=400, width=700,
        ):
            dpg.add_file_extension(".ini")
            dpg.add_file_extension(".*")

    def _open_config(self, sender, app_data, user_data):
        """
        Expects user_data to be the input_text id/tag.
        """
        cfgpathInput, parent = user_data
        cfgpath = dpg.get_value(cfgpathInput)
        if os.path.exists(cfgpath):
            # Then open it
            self.cfgs.append([
                DSPConfig(cfgpath),
                dpg.add_text(
                    cfgpath,
                    parent=parent
                )
            ])
        else:
            # Ask whether we want to create the config
            with dpg.window(
                label="Path does not exist",
                width=400, height=50,
                modal=True,
                show=True
            ) as popupModal:
                centreModal(popupModal)
                dpg.add_text("Would you like to create a new config at the specified path?")
                
                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="Yes",
                        callback=self._create_config,
                        user_data=[cfgpath, popupModal, parent]
                    )
                    dpg.add_button(
                        label="No",
                        callback=self._create_config,
                        user_data=[None, popupModal, parent]
                    )

        print(self.cfgs)

    def _create_config(self, sender, app_data, user_data):
        cfgpath, popupModal, parent = user_data
        # Create the new config
        if cfgpath is not None:
            self.cfgs.append([
                DSPConfig.new(cfgpath, allow_no_value=True),
                dpg.add_text(
                    cfgpath,
                    parent=parent
                )
            ])
        dpg.configure_item(popupModal, show=False)


