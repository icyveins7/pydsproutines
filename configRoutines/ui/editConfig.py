import dearpygui.dearpygui as dpg

from .._core import *

#%%
class EditConfigWindow:
    def __init__(self, cfg: DSPConfig, cfgpath: str):
        self.cfg = cfg
        self.cfgpath = cfgpath
        self._render()

    def _render(self):
        self.window = dpg.add_window(
            label=self.cfgpath,
            width=400, height=800, pos=(100, 100)
        )
        dpg.add_text("Test", parent=self.window)





