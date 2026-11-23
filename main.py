import tkinter as tk
from gui.app import AttendanceApp


def main():
    root = tk.Tk()
    root.title("Attendance System")
    root.geometry("1024x600")
    root.attributes("-fullscreen", True)

    app = AttendanceApp(root)

    root.bind("<Escape>", lambda event: app.on_close())

    root.mainloop()


if __name__ == "__main__":
    main()
