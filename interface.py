from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import os
import os.path
from waste_classifier import start
import cv2
import time
time.sleep(5)


def select_image():
    # grab a reference to the image panels
    global panelA, panelB

    # open a file chooser dialog and allow the user to select an input
    # image
    path = tkinter.filedialog.askopenfilename()
    save_path='testtrash'
    if len(path) > 0:
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        image = cv2.imread(path)
        p = os.path.join(save_path, "{}.jpg".format("test"))
        cv2.imwrite(p, image)
        cv2.waitKey(0)
        edged= start(1)
        os.remove(p)
        image = Image.fromarray(image)
        edged = Image.fromarray(edged)

        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        edged = ImageTk.PhotoImage(edged)

        if panelA is None or panelB is None:
            # the first panel will store our original image
            panelA = Label(image=image)
            panelA.image = image
            panelA.pack(side="left", padx=10, pady=10)

            # while the second panel will store the edge map
            panelB = Label(image=edged)
            panelB.image = edged
            panelB.pack(side="right", padx=10, pady=10)


            # otherwise, update the image panels
        else:
            # update the panels
            panelA.configure(image=image)
            panelB.configure(image=edged)
            panelA.image = image
            panelB.image = edged

root = Tk()
panelA = None
panelB = None
# create a button, then when pressed, will trigger a file chooser
# button the GUI

button = Button(root, text="Select an image", command=select_image)
button.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")


# kick off the GUI

root.mainloop()