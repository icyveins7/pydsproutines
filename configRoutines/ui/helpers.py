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