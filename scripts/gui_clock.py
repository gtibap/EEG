# for python 3.x use 'tkinter' rather than 'Tkinter'
import tkinter as tk
import time
from datetime import datetime

class App():
    def __init__(self):
        self.time_ini=datetime.now()
        self.time_now=datetime.now()
        
        self.flag_run = False
        self.flag_ce = False
        self.flag_oe = False
        self.flag_re = False
        
        self.time_diff=self.time_now - self.time_ini
        hh, mm, ss = self.time_conversion(self.time_diff)
        disp_time = str(mm).zfill(2)+':'+str(ss).zfill(2)
        
        # print(f'diff: {str(hh).zfill(2)}:{str(mm).zfill(2)}:{str(ss).zfill(2)}')
        # print(self.time_diff.days, self.time_diff.seconds, self.time_diff.microseconds )
        
        self.root = tk.Tk()
        self.root.title('Centre de Recherche, Hôpital du Sacré-Coeur de Montréal')
        self.root.geometry('640x480+300+300')
        self.root.resizable(False, False)
        
        self.ce_label = tk.Label(text=disp_time)
        self.oe_label = tk.Label(text=disp_time)
        self.re_label = tk.Label(text=disp_time)
        
        # self.ce_label.configure(text=disp_time)
        # self.oe_label.configure(text=disp_time)
        # self.re_label.configure(text=disp_time)
        
        self.ce_button = tk.Button(
                           text="Closed Eyes\nStart", 
                           fg="green",
                           command=self.ce_clock)
                           
        self.oe_button = tk.Button(
                           text="Opened Eyes\nStart", 
                           fg="blue",
                           command=self.oe_clock)
                           
        self.re_button = tk.Button(
                           text="Resting\nStart", 
                           fg="purple",
                           command=self.re_clock)
        
        self.root.grid()
        self.ce_label.grid(row=0, column=0)
        self.oe_label.grid(row=0, column=1)
        self.re_label.grid(row=0, column=2)
        self.ce_button.grid(row=1, column=0)
        self.oe_button.grid(row=1, column=1)
        self.re_button.grid(row=1, column=2)
        self.update_clock()
        self.root.mainloop()

    def update_clock(self):
        # now = time.strftime("%H:%M:%S")
        self.time_now=datetime.now()
        
        if self.flag_ce == True:
            disp_time = self.display_time(self.time_ce, self.time_now)
            self.ce_label.configure(text=disp_time)
        elif self.flag_oe == True:
            disp_time = self.display_time(self.time_oe, self.time_now)
            self.oe_label.configure(text=disp_time)
        elif self.flag_re == True:
            disp_time = self.display_time(self.time_re, self.time_now)
            self.re_label.configure(text=disp_time)
        
        # self.celabel.configure(text=disp_time)
        elif self.flag_run == True:
            pass
            
        else:
            pass
            
        self.root.after(1000, self.update_clock)
    
        
    def ce_clock(self):
        ## restart the clock
        self.time_ce = datetime.now()
        self.flag_ce = True
        self.flag_oe = False
        self.flag_re = False
        # self.update_clock()
        
    def oe_clock(self):
        ## restart the clock
        self.time_oe = datetime.now()
        self.flag_ce = False
        self.flag_oe = True
        self.flag_re = False
        
    def re_clock(self):
        ## restart the clock
        self.time_re = datetime.now()
        self.flag_ce = False
        self.flag_oe = False
        self.flag_re = True

    def display_time(self, time_ini, time_now):
        
        time_diff  = time_now - time_ini
        hh, mm, ss = self.time_conversion(time_diff)
        disp_time = str(mm).zfill(2)+':'+str(ss).zfill(2)
        
        return disp_time

    def time_conversion(self, duration):
        days, seconds = duration.days, duration.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        return hours, minutes, seconds
        

app=App()

