import dearpygui.dearpygui as dpg

from .._core import *

#%%
class EditConfigWindow:
    def __init__(self, cfg: DSPConfig):
        self.cfg = cfg

    def run(self):
        with dpg.window(
            label=dpg.get_value(self.cfg[1])
        ) as window:
            dpg.add_text("Test")


