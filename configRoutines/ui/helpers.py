import dearpygui.dearpygui as dpg

#%%
def centreModal(modalId):
    viewportHeight = dpg.get_viewport_height()
    viewportWidth = dpg.get_viewport_width()
    modalHeight = dpg.get_item_height(modalId)
    modalWidth = dpg.get_item_width(modalId)
    dpg.set_item_pos(
        modalId,
        [(viewportWidth - modalWidth) // 2,
        (viewportHeight - modalHeight) // 2]
    )

def getAppropriateInput(type, *args, **kwargs):
    """
    Calls the appropriate dpg.add_input_ function for the given type,
    and passes all args and kwargs.

    int: dpg.add_input_int
    float: dpg.add_input_float
    str: dpg.add_input_text
    bool: dpg.add_checkbox
    """
    inputWidgets = {
        int: dpg.add_input_int,
        float: dpg.add_input_double,
        str: dpg.add_input_text,
        bool: dpg.add_checkbox,
        list: TextInputWithComboWidget
    }
    return inputWidgets[type](*args, **kwargs)

def setValueIfNotNone(widget, value):
    if value is not None:
        dpg.set_value(widget, value)


###############################
class TextInputWithComboWidget:
    """
    Creates a text input and a dropdown next to it.
    The text input is the primary input widget, but the dropdown
    can be used to autofill the text input.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes object with two widgets:
        1) self.textInput: The text input widget
        2) self.dropdown: The dropdown widget

        User should set the list of values for the dropdown after init.
        """
        with dpg.group(horizontal=True):
            self.dropdown = dpg.add_combo(
                no_preview=True,
                callback=self.dropdown_callback,
                *args,
                **kwargs
            ) # Width=-1 only works if its the last one
            self.textInput = dpg.add_input_text(*args, **kwargs)
            

    def dropdown_callback(self, sender, app_data, user_data):
        dpg.set_value(self.textInput, dpg.get_value(sender))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class CheckboxEnabledWidget:
    """
    Contains a dpg.add_checkbox and another widget, built from getAppropriateInput.
    By default enables callback that shows the second widget based on checkbox value.

    You should subclass this and redefine the on_checkbox_changed method.
    In the new method, remember to call super().on_checkbox_changed(self, sender, app_data, user_data)
    in order to maintain the widget appear/disappear callback.
    """
    def __init__(self, type: type, *args, **kwargs):
        self.checkbox = dpg.add_checkbox(
            label="enabled?",
            callback=self.on_checkbox_changed            
        ) # Defaults to False

        # Create but hide widget
        self.widget = getAppropriateInput(type, *args, show=False, **kwargs)

    def on_checkbox_changed(self, sender, app_data, user_data):
        if isinstance(self.widget, TextInputWithComboWidget):
            dpg.configure_item(self.widget.textInput, show=dpg.get_value(self.checkbox))
            dpg.configure_item(self.widget.dropdown, show=dpg.get_value(self.checkbox))
                                                                         
        else:
            # Always enable/disable the widget depending on it
            dpg.configure_item(self.widget, show=dpg.get_value(self.checkbox))

    def set_value(self, enabled: bool, widgetValue=None, **kwargs):
        dpg.set_value(self.checkbox, enabled)

        # Special compound widgets
        if isinstance(self.widget, TextInputWithComboWidget):
            dpg.configure_item(self.widget.textInput, show=enabled)
            dpg.configure_item(self.widget.dropdown, show=enabled)
            if enabled:
                # Set value directly on text input
                setValueIfNotNone(self.widget.textInput, widgetValue)

        else: # Normal widgets
            dpg.configure_item(self.widget, show=enabled)
            if enabled:
                setValueIfNotNone(self.widget, widgetValue)

    def get_value(self):
        isChecked = dpg.get_value(self.checkbox)
        if isChecked:
            if isinstance(self.widget, TextInputWithComboWidget):
                value = dpg.get_value(self.widget.textInput)
            else:
                value = dpg.get_value(self.widget)
        else:
            value = None

        return (isChecked, value)
