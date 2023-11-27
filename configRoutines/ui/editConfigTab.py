import dearpygui.dearpygui as dpg

from .._core import *
from .helpers import getAppropriateInput, setValueIfNotNone
from .helpers import CheckboxEnabledWidget

class ConfigPairedWidget(CheckboxEnabledWidget):
    def __init__(self, enabled: bool, type: type, *args, **kwargs):
        if type != bool:
            super().__init__(enabled, type, *args, width=-1, **kwargs)
        else:
            super().__init__(enabled, type, *args, **kwargs)

    def on_checkbox_changed(self, sender, app_data, user_data):
        super().on_checkbox_changed(sender, app_data, user_data)



#%%
class EditConfigTab:
    def __init__(self, tabLabel: str, tab_bar: int, columns: list, cfg: DSPConfig):
        """
        Constructs a tab and adds it to the tab bar.
        Should not be constructed directly. 
        Subclass this and reimplement for the specific tab.

        Call _renderTab() in the ctor of derived class 
        with the appropriate section type.

        Parameters
        ----------
        tabLabel : str
            Label on the tab bar.
        tab_bar : int
            The parent widget, returned from tab_bar().
        columns : list
            List of tuples describing each field in the table.
            Recommended to subclass and then define this statically in the class definition.
            Then call super().__init__() in ctor with the specific subclass columns.

            First 2 rows are fixed.
            Example:
            [
                ('', None), # For buttons
                ('name', str), # Title of section
                ('field1', bool) # Example of first column 
                ...
            ]
        """
        self.tabLabel = tabLabel
        self.tab_bar = tab_bar
        self.columns = columns
        self.cfg = cfg

        # Container to store widgets for referencing
        self.widgets = {
            'rows': [],
            'cells': [],
            'table': None
        }

        # # Render it on init
        # self._renderTab()

    def _renderTab(self, content: dict, renderRows: bool=False):
        with dpg.tab(label=self.tabLabel, parent=self.tab_bar):
            # Create a table for the signals
            # Refer to SignalSectionProxy for details
            with dpg.table(
                resizable=True,
                # don't use row_background=True otherwise the alternating colours is not clear
                borders_outerH=True, borders_innerH=True,
                borders_innerV=True, borders_outerV=True
            ) as table:
                # Store this widget into container for reference later
                self.widgets['table'] = table

                for col in self.columns:
                    dpg.add_table_column(label=col[0])

                # Render rows if asked for
                if renderRows:
                    self._renderRows(content)

            # Row button adder
            dpg.add_button(
                label="Add Row",
                width=-1,
                callback=self._createRow
            )

    def _renderRows(self, content: dict, clearBefore: bool=False):
        if clearBefore:
            print(self.widgets['rows'])
            for row in self.widgets['rows']:
                dpg.delete_item(row)
            self.widgets['rows'].clear()
            # Clear the cells too
            self.widgets['cells'].clear()

        print("Rendering %s rows" % (self.tabLabel))
        for key, item in content.items():
            # Create the base row
            rowWidgets = self._createRow()
            # Set the values of the row
            self._setRowValues(key, item, rowWidgets)

    def _createRow(self):
        rowWidgets = dict()
        with dpg.table_row(parent=self.widgets['table']) as row:
            # Save rows for reference
            try:
                self.widgets['rows'].append(row)
            except KeyError:
                self.widgets['rows'] = [row]  

            # Make cell for buttons
            with dpg.table_cell(parent=row) as cell:
                deleteBtn = dpg.add_button(
                    label="Delete",
                    callback=self._deleteRow
                    # Set user_data at the end
                )
                # dpg.bind_item_theme(deleteBtn, self.delBtn_theme)
                rowWidgets['deleteBtn'] = deleteBtn
                dupBtn = dpg.add_button(
                    label="Duplicate",
                    callback=self._duplicateRow
                    # Set user_data at the end
                )
                # dpg.bind_item_theme(dupBtn, self.dupBtn_theme)
                rowWidgets['dupBtn'] = dupBtn

            # Make cell for name first, always there
            with dpg.table_cell(parent=row) as cell:
                inputWidget = getAppropriateInput(str, width=-1, parent=cell)
                rowWidgets['name'] = inputWidget

            # Make paired widgets for each column
            for col in self.columns:
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
            self.widgets['cells'].append(rowWidgets)
        except KeyError:
            self.widgets['cells'] = [rowWidgets]
        
        return rowWidgets


    def _setRowValues(self, name: str, content: SectionProxy, rowWidgets: dict):
        # Reference the order of the columns given at class definition
        # Ignore the button column for obvious reasons
        # Set the values of the row
        dpg.set_value(rowWidgets['name'], name)
        for col in self.columns:
            key, castType = col
            if key == 'name' or key == '': # Skip name and button column
                continue
            # If key exists, we enable it
            try:
                rawval = content.get(key)
                enabled = False if rawval is None else True
                val = castType(rawval)
            except (KeyError, TypeError):
                val = None
                enabled = False

            rowWidgets[key].trigger_enabled(enabled, val)

    def _writeToConfig(self, content: dict, cfg: DSPConfig):
        """
        Amends the internal signal-only structs of the DSPConfig object.
        Does not dump this to file; that is to be called separately.
        """
        pass # TODO, this section probably needs to be overloaded in the derived classes
        # existingKeys = list(content.keys())
        # # Read each row of the widgets
        # for cellRow in self.widgets['cells']:
        #     # Check if the name exists
        #     name = dpg.get_value(cellRow['name'])

        #     # Create new one if it doesn't
        #     if name not in existingKeys:
        #         cfg.addSignal(name)
        #     # Mark those that have been found
        #     else: # else-statement so that it will not try to remove if it's not present at first
        #         existingKeys.remove(name)
            
        #     # Then amend the section internally
        #     for key, _ in self.columns:
        #         if key == 'name' or key == '':
        #             continue

        #         # If enabled then set it config
        #         if dpg.get_value(cellRow[key].checkbox):
        #             cfg.getSig(name)[key] = str(dpg.get_value(
        #                 cellRow[key].widget
        #             ))
        #         else: # Otherwise remove it
        #             if key in cfg.getSig(name):
        #                 cfg.getSig(name).pop(key)

        # # Finally, remove those that no longer exist
        # # These are the remainders from existingKeys
        # for name in existingKeys:
        #     cfg.removeSignal(name)
    
    def _deleteRow(self, sender, app_data, user_data):
        """
        Callback to delete a row and remove 
        its associated widgets from all containers.

        Parameters
        ----------
        user_data: list
            Should contain [row, rowWidgets],
            where row is the dpg.table_row widget
            and rowWidgets is a dictionary of widgets.
            See _createRow for details.
        """
        row, rowWidgets = user_data
        dpg.delete_item(row) # Delete the row of widgets
        # And remove from the internal containers
        self.widgets['rows'].remove(row)
        self.widgets['cells'].remove(rowWidgets)

    def _duplicateRow(self, sender, app_data, user_data):
        """
        Callback to duplicate a row.
        All columns should be identical, except the name column,
        which has '_copy' appended to it.
        """
        row, rowWidgets = user_data
        # Create a new row
        newRowWidgets = self._createRow()
        # Set the values by extracting them from the UI
        # We cannot extract from the config because the row may have been changed and
        # we haven't flushed the changes to the file yet
        content = dict()
        for col in self.columns:
            key, castType = col
            if key == 'name' or key == '': # Skip name and button column
                continue
            # Check the original rowWidget
            if dpg.get_value(rowWidgets[key].checkbox):
                content[key] = dpg.get_value(rowWidgets[key].widget)

        # Set the new widget values
        self._setRowValues(
            dpg.get_value(rowWidgets['name']) + "_copy",
            content,
            newRowWidgets
        )
        # No need to save them as we saved them during creation

    
#%%
class EditSourcesTab(EditConfigTab):
    columns = [
        ('', None),
        ('name', str),
        ('srcdir', str),
        ('fs', float),
        ('fc', float),
        ('conjSamples', bool),
        ('headerBytes', int),
        ('dtype', str),
        ('lonlatalt', str)
    ]

    def __init__(self, tab_bar: int, cfg: DSPConfig):
        super().__init__("Sources", tab_bar, self.columns, cfg)
        self._renderTab(self.cfg.allSources)