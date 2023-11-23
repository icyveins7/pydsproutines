import dearpygui.dearpygui as dpg

from .._core import *
from .helpers import getAppropriateInput, setValueIfNotNone
from .helpers import CheckboxEnabledWidget

#%%
class ConfigPairedWidget(CheckboxEnabledWidget):
    def __init__(self, enabled: bool, type: type, *args, **kwargs):
        if type != bool:
            super().__init__(enabled, type, *args, width=-1, **kwargs)
        else:
            super().__init__(enabled, type, *args, **kwargs)

    def on_checkbox_changed(self, sender, app_data, user_data):
        super().on_checkbox_changed(sender, app_data, user_data)
        print('hello')


#%%
class EditConfigWindow:
    signalColumns = [
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

    def __init__(self, cfg: DSPConfig, cfgpath: str, openCfgObject):
        self.cfg = cfg
        self.cfgpath = cfgpath
        self.openCfgObject = openCfgObject
        
        # Containers for widgets
        self.signalWidgets = dict()

        # Render on init
        self._initialRender()

    def _initialRender(self):
        self.window = dpg.add_window(
            label=self.cfgpath,
            width=400, height=800, pos=(100, 100),
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
        self._renderSignalsTab(renderRows=True)
        self._renderSourcesTab()
        self._renderProcessesTab()
        self._renderWorkspacesTab()

    def _writeChanges(self):
        pass

    def _resetConfig(self):
        # Reload the config from the file
        self.cfg = DSPConfig(self.cfgpath)

        # Re-render everything
        self._renderSignalsRows(clearBefore=True)


    def _renderSignalsTab(self, renderRows: bool=False):
        with dpg.tab(label="Signals", parent=self.tab_bar):
            # Create a table for the signals
            # Refer to SignalSectionProxy for details
            with dpg.table(
                resizable=True,
                row_background=True,
                borders_outerH=True, borders_innerH=True,
                borders_innerV=True, borders_outerV=True
            ) as table:
                # Store this widget into container for reference later
                self.signalWidgets['table'] = table

                for col in self.signalColumns:
                    dpg.add_table_column(label=col[0])

                # Render rows if asked for
                if renderRows:
                    self._renderSignalsRows()

            # Row button adder
            dpg.add_button(
                label="Add Row",
                width=-1,
                callback=self._createSignalRow
            )

    def _createSignalRow(self):
        rowWidgets = dict()
        with dpg.table_row(parent=self.signalWidgets['table']) as row:
            # Save rows for reference
            try:
                self.signalWidgets['rows'].append(row)
            except KeyError:
                self.signalWidgets['rows'] = [row]  

            # Make cell for name first, always there
            with dpg.table_cell(parent=row) as cell:
                inputWidget = getAppropriateInput(str, width=-1, parent=cell)
                rowWidgets['name'] = inputWidget

            # Make paired widgets for each column
            for col in self.signalColumns:
                key, castType = col
                if key == 'name': # Skip name column
                    continue

                with dpg.table_cell(parent=row):
                    cpw = ConfigPairedWidget(
                        False, castType
                    )
                    rowWidgets[key] = cpw
        
        return rowWidgets

            
    
    def _renderSignalsRows(self, clearBefore: bool=False):
        if clearBefore:
            print(self.signalWidgets['rows'])
            for row in self.signalWidgets['rows']:
                dpg.delete_item(row)
            self.signalWidgets['rows'].clear()
        # Load all signals from the config
        signals = self.cfg.allSignals
        print("Rendering signal rows")
        for signalName, signal in signals.items():
            # Create the base row
            rowWidgets = self._createSignalRow()
            # Set the values of the row
            dpg.set_value(rowWidgets['name'], signalName)
            for col in self.signalColumns:
                key, castType = col
                if key == 'name': # Skip name column
                    continue
                # If key exists, we enable it
                try:
                    rawval = signal.get(key)
                    enabled = False if rawval is None else True
                    val = castType(rawval)
                except (KeyError, TypeError):
                    val = None
                    enabled = False

                rowWidgets[key].trigger_enabled(enabled, val)
                     
    def _renderSourcesTab(self):
        with dpg.tab(label="Sources", parent=self.tab_bar):
            sources = self.cfg.allSources
            # Create a table for the sources
            # Refer to SourceSectionProxy for details
            with dpg.table():
                columns = [
                    ('name', str),
                    ('srcdir', str),
                    ('fs', float),
                    ('fc', float),
                    ('conjSamples', bool),
                    ('headerBytes', int),
                    ('dtype', str),
                    ('lonlatalt', str)
                ]
                for col in columns:
                    dpg.add_table_column(label=col[0])

                for sourceName, source in sources.items():
                    with dpg.table_row():
                        # Make cell for name first, always there
                        with dpg.table_cell():
                            inputWidget = getAppropriateInput(str, width=-1)
                            dpg.set_value(inputWidget, sourceName)

                        # Make paired widgets for each column
                        for col in columns:
                            key, castType = col
                            if key == 'name': # Skip name column
                                continue

                            with dpg.table_cell():
                                # If key exists, we enable it
                                try:
                                    rawval = source.get(key)
                                    enabled = False if rawval is None else True
                                    val = castType(rawval)
                                except (KeyError, TypeError):
                                    val = None
                                    enabled = False

                                cpw = ConfigPairedWidget(
                                    enabled, castType
                                )
                                setValueIfNotNone(cpw.widget, val)

    def _renderProcessesTab(self):
        with dpg.tab(label="Processes", parent=self.tab_bar):
            w = ConfigPairedWidget(False, str)
            

    def _renderWorkspacesTab(self):
        with dpg.tab(label="Workspaces", parent=self.tab_bar):
            pass




# %%

