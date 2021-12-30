# %%
import PySimpleGUI as sg
import os.path

# window layout in 2 columns

# nested list of elements representing a vertical column of UI
# creates a browse button to find image folder desired
file_list_column = [
    [
        sg.Text("GAN Generated Images"),
        # key parameter is to identify access to the element
        # turn on or off events for each element using enable_events
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(initial_folder='C:/Users/burro/Desktop/GUI_test/'),
    ],
    [
        # listbox displays a list of paths to images that you can choose from to display to user
        # it can be prefilled with values by passing in a list of strings
        sg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-")
    ],
]

# right hand column of elements
# creates three elements
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(40, 1), key="-TOUT-")],  # displays name of file
    [sg.Image(key="-IMAGE-")],  # displays image selected by user
]

# Define Layout
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeparator(),  # vertical separator
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)

# event loop = logic of program
# extract events and values from window
# event = the key string of which element is chosen by user to interact
# values is python dict mapping key to value e.g. if user selects a folder, "-FOLDER-" is mapped to folder path
# conditional statements are used to control what happens
# if event = exit or the user closes the window, the loop is broken
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break

# first part of next conditional statement in loop
# check event against FOLDER key
# if event exists, then user has chosen right folder, os.listdir to get the files
# filter list of files to .png and .gif only
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
        # get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith(".png")
        ]
        window["-FILE LIST-"].update(fnames)

# next part of conditional statement
# if the event = FILE LIST, user has chosen a file in the listbox and image() and text() elements should be updated
    elif event == "-FILE LIST-":  # a file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)
            window["-IMAGE-"].update(filename=filename)
        except:
            pass

window.close()
