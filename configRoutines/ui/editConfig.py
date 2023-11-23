import dearpygui.dearpygui as dpg

from .._core import *
from .helpers import getAppropriateInput, setValueIfNotNone
from .helpers import CheckboxEnabledWidget

#%%
class ConfigPairedWidget(CheckboxEnabledWidget):
    def on_checkbox_changed(self, sender, app_data, user_data):
        super().on_checkbox_changed(sender, app_data, user_data)
        print('hello')


#%%
class EditConfigWindow:
    def __init__(self, cfg: DSPConfig, cfgpath: str, openCfgObject):
        self.cfg = cfg
        self.cfgpath = cfgpath
        self.openCfgObject = openCfgObject
        self._initialRender()

    def _initialRender(self):
        self.window = dpg.add_window(
            label=self.cfgpath,
            width=400, height=800, pos=(100, 100),
            # Set cleanup to the same one as the external button
            on_close=self.openCfgObject._del_editConfig,
            user_data=[self.cfgpath]
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
                    ('name', str),
                    ('target_fc', float),
                    ('baud', float),
                    ('numPeriodBits', int),
                    ('numBurstBits', int),
                    ('numGuardBits', int),
                    ('numBursts', int),
                    ('hasChannels', bool),
                    ('channelSpacingHz', float)
                ]
                for col in columns:
                    dpg.add_table_column(label=col[0])

                for signalName, signal in signals.items():
                    with dpg.table_row():
                        # Make cell for name first, always there
                        with dpg.table_cell():
                            inputWidget = getAppropriateInput(str, width=-1)
                            dpg.set_value(inputWidget, signalName)

                        # Make paired widgets for each column
                        for col in columns:
                            key, castType = col
                            with dpg.table_cell():
                                # If key exists, we enable it
                                try:
                                    val = castType(signal.get(key))
                                    enabled = True
                                except KeyError:
                                    val = None
                                    enabled = False

                                cpw = ConfigPairedWidget(
                                    enabled, castType, width=-1
                                )
                                setValueIfNotNone(cpw.widget, val)
                
                
    def _renderSourcesTab(self):
        with dpg.tab(label="Sources", parent=self.tab_bar):
            sources = self.cfg.allSources
            # Create a table for the sources
            # Refer to SourceSectionProxy for details
            with dpg.table():
                columns = [
                    'name',
                    'srcdir',
                    'fs',
                    'fc',
                    'conjSamples',
                    'headerBytes',
                    'dtype'
                    'lonlatalt'
                ]
                for col in columns:
                    dpg.add_table_column(label=col)

                for sourceName, source in sources.items():
                    with dpg.table_row():
                        # Add cell for name
                        with dpg.table_cell():
                            inputWidget = dpg.add_input_text(width=-1)
                            dpg.set_value(inputWidget, sourceName)
                        # Add cell for srcdir
                        with dpg.table_cell():
                            inputWidget = dpg.add_input_text(width=-1)
                            dpg.set_value(inputWidget, source.srcdir)
                        # Add cell for fs
                        with dpg.table_cell():
                            inputWidget = dpg.add_input_text(width=-1)
                            dpg.set_value(inputWidget, source.fs)
                        # Add cell for fc
                        with dpg.table_cell():
                            inputWidget = dpg.add_input_text(width=-1)
                            dpg.set_value(inputWidget, source.fc)
                        # Add cell for conjSamples
                        with dpg.table_cell():
                            inputWidget = dpg.add_input_text(width=-1)
                            dpg.set_value(inputWidget, source.conjSamples)
                        # Add cell for headerBytes
                        with dpg.table_cell():
                            inputWidget = dpg.add_input_text(width=-1)
                            dpg.set_value(inputWidget, source.headerBytes)
                        # Add cell for dtype
                        with dpg.table_cell():
                            inputWidget = dpg.add_input_text(width=-1)
                            dpg.set_value(inputWidget, source.dtype)
                        # Add cell for lonlatalt
                        with dpg.table_cell():
                            inputWidget = dpg.add_input_text(width=-1)
                            dpg.set_value(inputWidget, source.lonlatalt)
                            

    def _renderProcessesTab(self):
        with dpg.tab(label="Processes", parent=self.tab_bar):
            w = ConfigPairedWidget(False, str)
            

    def _renderWorkspacesTab(self):
        with dpg.tab(label="Workspaces", parent=self.tab_bar):
            pass




