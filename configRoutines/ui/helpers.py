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
    inputWidgets = {
        int: dpg.add_input_int,
        float: dpg.add_input_double,
        str: dpg.add_input_text,
        bool: dpg.add_checkbox
    }
    return inputWidgets[type](*args, **kwargs)

def setValueIfNotNone(widget, value):
    if value is not None:
        dpg.set_value(widget, value)