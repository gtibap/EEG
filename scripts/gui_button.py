import tkinter as tk
from datetime import datetime
import pandas as pd

list_opened_eyes = []
list_closed_eyes = []

path = '../data/'
prefix = 'test'
filename_opened_eyes = 'opened_eyes.txt'
filename_closed_eyes = 'closed_eyes.txt'

def writeList(filename, lines):
    with open(path+filename, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")
    return 0
    
   
def eyesClosed():
    now = datetime.now().strftime('%H:%M:%S')
    list_closed_eyes.append(now)
    writeList(prefix+filename_closed_eyes, list_closed_eyes)
    # print(f'eyes closed at {now}')
    
    return 0

def eyesOpened():
    now = datetime.now().strftime('%H:%M:%S')
    list_opened_eyes.append(now)
    writeList(prefix+filename_opened_eyes, list_opened_eyes)
    # print(f'eyes opened at {now}')
    return 0


def main(args):
    global prefix
    
    prefix = args[1]
    print(f'patient {prefix}')

    root = tk.Tk()
    frame = tk.Frame(root)
    frame.pack()

    button = tk.Button(frame, 
                       text="Eyes closed", 
                       fg="blue",
                       command=eyesClosed)
    button.pack(side=tk.LEFT)
    slogan = tk.Button(frame,
                       text="Eyes opened",
                       fg="green",
                       command=eyesOpened)
    slogan.pack(side=tk.LEFT)

    root.mainloop()

    
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
