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


#%%
class EditConfigWindow:
    signalColumns = [
        ('', None), # Button column
        ('name', str), # Does not have a checkbox enabler
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

        # Instantiate the themes to be used
        self._createButtonThemes()

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
        self._renderSignalsTab(renderRows=True)
        self._renderSourcesTab()
        self._renderProcessesTab()
        self._renderWorkspacesTab()


    def _createButtonThemes(self):
        with dpg.theme() as delBtn_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (255, 0, 0), category=dpg.mvThemeCat_Core)

            self.delBtn_theme = delBtn_theme

        with dpg.theme() as dupBtn_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 255), category=dpg.mvThemeCat_Core)

            self.dupBtn_theme = dupBtn_theme


    def _writeChanges(self):
        self._writeSignals()

        # Dump to file
        with open(self.cfgpath, 'w') as cfgfile:
            self.cfg.write(cfgfile)

    def _resetConfig(self):
        # Reload the config from the file
        self.cfg = DSPConfig(self.cfgpath)

        # Re-render everything
        self._renderSignalsRows(clearBefore=True)

    ####################### Signals ########################
    def _writeSignals(self):
        """
        Amends the internal signal-only structs of the DSPConfig object.
        Does not dump this to file; that is to be called separately.
        """
        existingSignals = list(self.cfg.allSignals.keys())
        # Read each row of the widgets
        for cellRow in self.signalWidgets['cells']:
            # Check if the name exists
            signalName = dpg.get_value(cellRow['name'])

            # Create new one if it doesn't
            if signalName not in existingSignals:
                self.cfg.addSignal(signalName)
            # Mark those that have been found
            else: # else-statement so that it will not try to remove if it's not present at first
                existingSignals.remove(signalName)
            
            # Then amend the section internally
            for key, _ in self.signalColumns:
                if key == 'name' or key == '':
                    continue

                # If enabled then set it config
                if dpg.get_value(cellRow[key].checkbox):
                    self.cfg.getSig(signalName)[key] = str(dpg.get_value(
                        cellRow[key].widget
                    ))
                else: # Otherwise remove it
                    if key in self.cfg.getSig(signalName):
                        self.cfg.getSig(signalName).pop(key)


        # Finally, remove those that no longer exist
        # These are the remainders from existingSignals
        for signalName in existingSignals:
            self.cfg.removeSignal(signalName)

    def _renderSignalsTab(self, renderRows: bool=False):
        with dpg.tab(label="Signals", parent=self.tab_bar):
            # Create a table for the signals
            # Refer to SignalSectionProxy for details
            with dpg.table(
                resizable=True,
                # don't use row_background=True otherwise the alternating colours is not clear
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

            # Make cell for buttons
            with dpg.table_cell(parent=row) as cell:
                deleteBtn = dpg.add_button(
                    label="Delete",
                    callback=self._deleteSignalRow
                    # Set user_data at the end
                )
                dpg.bind_item_theme(deleteBtn, self.delBtn_theme)
                rowWidgets['deleteBtn'] = deleteBtn
                dupBtn = dpg.add_button(
                    label="Duplicate",
                    callback=self._duplicateSignalRow
                    # Set user_data at the end
                )
                dpg.bind_item_theme(dupBtn, self.dupBtn_theme)
                rowWidgets['dupBtn'] = dupBtn

            # Make cell for name first, always there
            with dpg.table_cell(parent=row) as cell:
                inputWidget = getAppropriateInput(str, width=-1, parent=cell)
                rowWidgets['name'] = inputWidget

            # Make paired widgets for each column
            for col in self.signalColumns:
                key, castType = col
                if key == 'name' or key == '': # Skip button and name column
                    continue

                with dpg.table_cell(parent=row):
                    cpw = ConfigPairedWidget(
                        False, castType
                    )
                    rowWidgets[key] = cpw
        
            # Attach entire row to buttons' user data
            dpg.set_item_user_data(
                deleteBtn, [row, rowWidgets]
            )
            dpg.set_item_user_data(
                dupBtn, [row, rowWidgets]
            )

        try:
            self.signalWidgets['cells'].append(rowWidgets)
        except KeyError:
            self.signalWidgets['cells'] = [rowWidgets]
        
        return rowWidgets

    def _deleteSignalRow(self, sender, app_data, user_data):
        """
        Callback to delete a signal row and remove 
        its associated widgets from all containers.

        Parameters
        ----------
        user_data: list
            Should contain [row, rowWidgets],
            where row is the dpg.table_row widget
            and rowWidgets is a dictionary of widgets.
            See _createSignalRow for details.
        """
        row, rowWidgets = user_data
        dpg.delete_item(row) # Delete the row of widgets
        # And remove from the internal containers
        self.signalWidgets['rows'].remove(row)
        self.signalWidgets['cells'].remove(rowWidgets)

    def _duplicateSignalRow(self, sender, app_data, user_data):
        row, rowWidgets = user_data
        # Create a new row
        newRowWidgets = self._createSignalRow()
        # Set the values by extracting them from the UI
        # We cannot extract from the config because the row may have been changed and
        # we haven't flushed the changes to the file yet
        signal = dict()
        for col in self.signalColumns:
            key, castType = col
            if key == 'name' or key == '': # Skip name and button column
                continue
            # Check the original rowWidget
            if dpg.get_value(rowWidgets[key].checkbox):
                signal[key] = dpg.get_value(rowWidgets[key].widget)

        # Set the new widget values
        self._setSignalRowValues(
            dpg.get_value(rowWidgets['name']) + "_copy",
            signal,
            newRowWidgets
        )
        # No need to save them as we saved them during creation

    def _setSignalRowValues(self, signalName: str, signal: dict, rowWidgets: dict):
        # Reference the order of the columns given at class definition
        # Ignore the button column for obvious reasons
        # Set the values of the row
        dpg.set_value(rowWidgets['name'], signalName)
        for col in self.signalColumns:
            key, castType = col
            if key == 'name' or key == '': # Skip name and button column
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

    
    def _renderSignalsRows(self, clearBefore: bool=False):
        if clearBefore:
            print(self.signalWidgets['rows'])
            for row in self.signalWidgets['rows']:
                dpg.delete_item(row)
            self.signalWidgets['rows'].clear()
            # Clear the cells too
            self.signalWidgets['cells'].clear()
        # Load all signals from the config
        signals = self.cfg.allSignals
        print("Rendering signal rows")
        for signalName, signal in signals.items():
            # Create the base row
            rowWidgets = self._createSignalRow()
            # Set the values of the row
            self._setSignalRowValues(signalName, signal, rowWidgets)


    ##################### Sources ########################
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

