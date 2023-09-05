import dearpygui.dearpygui as dpg

#%%
class ConfigEditor:
    def run(self):
        # Show a window that lets you open a config file
        with dpg.window(label="Open a config file", 
                        width=700, height=150,
                        no_close=True) as window:
            with dpg.group(horizontal=True):
                cfgPathInput = dpg.add_input_text(
                    label="Config filepath"
                )
                dpg.add_button(
                    label="Open",
                    callback=self.open_config,
                    user_data=cfgPathInput
                )

    def open_config(self, sender, app_data, user_data):
        with dpg.file_dialog(
            callback=lambda dialog, dialog_data: [dpg.set_value(
                user_data, # This is the tag for the path input passed in
                item # This is the key for the path selected
            ) for key, item in dialog_data['selections'].items()],
            # Although we use a list comprehension, we assume only 1 item, so it reads the last one
            height=400, width=700
        ):
            dpg.add_file_extension(".ini")
            dpg.add_file_extension(".*")
        
