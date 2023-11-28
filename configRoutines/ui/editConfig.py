import dearpygui.dearpygui as dpg

from .._core import *
from .helpers import getAppropriateInput, setValueIfNotNone
from .helpers import CheckboxEnabledWidget
from .editConfigTab import EditSignalsTab, EditSourcesTab, EditProcessesTab, EditWorkspacesTab

#%%
class ConfigPairedWidget(CheckboxEnabledWidget):
    def __init__(self, enabled: bool, type: type, *args, **kwargs):
        if type != bool:
            super().__init__(enabled, type, *args, width=-1, **kwargs)
        else:
            super().__init__(enabled, type, *args, **kwargs)

    def on_checkbox_changed(self, sender, app_data, user_data):
        super().on_checkbox_changed(sender, app_data, user_data)


#%%
class EditConfigWindow:
    def __init__(self, cfg: DSPConfig, cfgpath: str, openCfgObject):
        self.cfg = cfg
        self.cfgpath = cfgpath
        self.openCfgObject = openCfgObject

        # Render on init
        self._initialRender()
  

    def _initialRender(self):
        self.window = dpg.add_window(
            label=self.cfgpath,
            width=800, height=400, pos=(100, 100),
            # Set cleanup to the same one as the external button
            on_close=self.openCfgObject._del_editConfig,
            user_data=[self.cfgpath]
        )

        # Buttons for controls
        with dpg.group(horizontal=True, parent=self.window):
            dpg.add_button(
                label="Write Changes",
                callback=self._writeChanges
            )
            dpg.add_button(
                label="Reset",
                callback=self._resetConfig
            )
        dpg.add_separator(parent=self.window)
        
        self.tab_bar = dpg.add_tab_bar(parent=self.window)
        # Add the different tab objects
        self.signalsTab = EditSignalsTab(self.tab_bar, self.cfg)
        self.sourcesTab = EditSourcesTab(self.tab_bar, self.cfg)
        self.processesTab = EditProcessesTab(self.tab_bar, self.cfg)
        self.workspacesTab = EditWorkspacesTab(self.tab_bar, self.cfg)
        

    def _writeChanges(self):
        # Call writers for each type
        self.signalsTab._writeToConfig()
        self.sourcesTab._writeToConfig()
        self.processesTab._writeToConfig()
        # TODO: workspacesTab

        # Dump to file
        with open(self.cfgpath, 'w') as cfgfile:
            self.cfg.write(cfgfile)

    def _resetConfig(self):
        # Reload the config from the file
        self.cfg = DSPConfig(self.cfgpath)

        # Re-render everything
        self.signalsTab._renderRows(self.cfg.allSignals, clearBefore=True)
        self.sourcesTab._renderRows(self.cfg.allSources, clearBefore=True)
        self.processesTab._renderRows(self.cfg.allProcesses, clearBefore=True)
        # TODO: workspacesTab

    