import dearpygui.dearpygui as dpg

from .._core import *

#%%
class EditConfigWindow:
    def __init__(self, cfg: DSPConfig, cfgpath: str):
        self.cfg = cfg
        self.cfgpath = cfgpath
        self._initialRender()

    def _initialRender(self):
        self.window = dpg.add_window(
            label=self.cfgpath,
            width=400, height=800, pos=(100, 100)
        )
        
        self.tab_bar = dpg.add_tab_bar(parent=self.window)
        self._renderSignalsTab()
        self._renderSourcesTab()
        self._renderProcessesTab()
        self._renderWorkspacesTab()

    def _renderSignalsTab(self):
        with dpg.tab(label="Signals", parent=self.tab_bar):
            signals = self.cfg.allSignals
            # Create a table for the signals
            # Refer to SignalSectionProxy for details
            with dpg.table():
                columns = [
                    'target_fc',
                    'baud',
                    'numPeriodBits',
                    'numBurstBits',
                    'numGuardBits',
                    'numBursts',
                    'hasChannels',
                    'channelSpacingHz'
                ]
                for col in columns:
                    dpg.add_table_column(label=col)
                
                
    def _renderSourcesTab(self):
        with dpg.tab(label="Sources", parent=self.tab_bar):
            pass

    def _renderProcessesTab(self):
        with dpg.tab(label="Processes", parent=self.tab_bar):
            pass

    def _renderWorkspacesTab(self):
        with dpg.tab(label="Workspaces", parent=self.tab_bar):
            pass




