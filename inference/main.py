from mainWindow.MainWindow import Application
import tkinter as tk

# TODO: save image in db and display it in frontend
# TODO: change saving folder from real_time_data to assert

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(root)
    app.mainloop()
