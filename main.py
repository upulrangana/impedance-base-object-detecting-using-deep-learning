from tkinter import *
from tkinter import messagebox

from titlecase import titlecase

from core import get_prediction


def clicked():
    answer = get_prediction()
    messagebox.showinfo('Predicted Answer', titlecase(answer))


window = Tk()
window.configure(background="blue")
window.title("Impedance Bace Object Detecting")
window.geometry('350x200')
lbl = Label(window, text="Predict Ball", wraplength=300, background="green")
lbl.pack(fill=BOTH, expand=1)

btn = Button(window, text="Click Me", command=clicked)
btn.pack(fill=BOTH, padx=40, pady=20)
window.mainloop()
