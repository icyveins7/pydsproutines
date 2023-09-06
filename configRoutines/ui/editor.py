import dearpygui.dearpygui as dpg

from .openConfig import OpenConfigDialog

#%%
class ConfigEditor:
    def __init__(self):
        self.openConfig = OpenConfigDialog()

    def run(self):
        print(self.openConfig.run())
        
