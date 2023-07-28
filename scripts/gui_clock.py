# for python 3.x use 'tkinter' rather than 'Tkinter'
import tkinter as tk
import time
from datetime import datetime
import os

class App():
    def __init__(self):
        self.time_ini=datetime.now()
        self.time_now=datetime.now()
        
        self.flag_ini = False
        self.flag_end = False
        #self.flag_run = False
        self.flag_ce = False
        self.flag_oe = False
        self.flag_re = False
        
        self.time_diff=self.time_now - self.time_ini
        hh, mm, ss = self.time_conversion(self.time_diff)
        disp_time_0 = str(hh).zfill(2)+':'+str(mm).zfill(2)+':'+str(ss).zfill(2)
        disp_time_1 = str(mm).zfill(2)+':'+str(ss).zfill(2)
        
        # print(f'diff: {str(hh).zfill(2)}:{str(mm).zfill(2)}:{str(ss).zfill(2)}')
        # print(self.time_diff.days, self.time_diff.seconds, self.time_diff.microseconds )
        
        self.root = tk.Tk()
        # self.root.title('Centre de Recherche, Hôpital du Sacré-Coeur de Montréal')
        self.root.title('EEG Acquisition')
        self.root.geometry('480x240+300+300')
        self.root.resizable(False, False)
        self.id_var=tk.StringVar()
        self.section_var = tk.StringVar(None, 'A')
        
        self.id_label = tk.Label(text='ID Participant:')
        self.id_input = tk.Entry(textvariable =  self.id_var)
        
        self.ini_label = tk.Label(text=disp_time_0)
        self.end_label = tk.Label(text=disp_time_0)
        self.ce_label = tk.Label(text=disp_time_1, font=("Arial", 16))
        self.oe_label = tk.Label(text=disp_time_1, font=("Arial", 16))
        self.re_label = tk.Label(text=disp_time_1, font=("Arial", 16))
        
        # self.ce_label.configure(text=disp_time)
        # self.oe_label.configure(text=disp_time)
        # self.re_label.configure(text=disp_time)
        
        self.rb_resting = tk.Radiobutton(text='Section A: Resting', variable=self.section_var, value='A', command=self.print_selection)
        
        self.rb_biking = tk.Radiobutton(text='Section B: Biking', variable=self.section_var, value='B', command=self.print_selection)
    
        
        self.ini_button = tk.Button(
                           text="Start\nrecording", 
                           fg="green",
                           command=self.ini_clock)
        
        self.end_button = tk.Button(
                           text="End\nrecording", 
                           fg="red",
                           command=self.end_clock)
        
        self.ce_button = tk.Button(
                           text="Closed Eyes\nStart", 
                           fg="green",
                           command=self.ce_clock)
                           
        self.oe_button = tk.Button(
                           text="Opened Eyes\nStart", 
                           fg="blue",
                           command=self.oe_clock)
                           
        self.re_button = tk.Button(
                           text="Pause\nStart", 
                           fg="purple",
                           command=self.re_clock)

        self.final_label0 = tk.Label(text='')
        
        self.final_label1 = tk.Label(text='Centre de Recherche,')
        self.final_label2 = tk.Label(text='Hôpital du Sacré-Coeur')
        self.final_label3 = tk.Label(text='de Montréal')
        
        self.root.grid()
        self.id_label.grid(row=0, column=0)
        self.id_input.grid(row=0, column=1)
        self.ini_button.grid(row=0, column=2)
        self.end_button.grid(row=0, column=3)
        
        self.rb_resting.grid(row=1, column=1)
        self.ini_label.grid(row=1, column=2)
        self.end_label.grid(row=1, column=3)
        
        
        self.rb_biking.grid(row=2, column=1)
        
        self.ce_label.grid(row=3, column=0)
        self.oe_label.grid(row=3, column=1)
        self.re_label.grid(row=3, column=2)

        self.ce_button.grid(row=4, column=0)
        self.oe_button.grid(row=4, column=1)
        self.re_button.grid(row=4, column=2)
        
        self.final_label0.grid(row=5, column=0)
        
        self.final_label1.grid(row=6, column=0)
        self.final_label2.grid(row=6, column=1)
        self.final_label3.grid(row=6, column=2)
        
        
        self.update_clock()
        self.root.mainloop()

    def update_clock(self):
        # now = time.strftime("%H:%M:%S")
        self.time_now=datetime.now()
        
        if self.flag_ini == True:
            
            disp_time = self.display_time(self.time_ini, self.time_now)
            self.end_label.configure(text=disp_time)

            if self.flag_ce == True:
                disp_time = self.display_time(self.time_ce, self.time_now)
                self.ce_label.configure(text=disp_time[3:])
            elif self.flag_oe == True:
                disp_time = self.display_time(self.time_oe, self.time_now)
                self.oe_label.configure(text=disp_time[3:])
            elif self.flag_re == True:
                disp_time = self.display_time(self.time_re, self.time_now)
                self.re_label.configure(text=disp_time[3:])
                
            self.root.after(1000, self.update_clock)
        
        else:
            pass
            
        if self.flag_end == True:
            self.flag_ini = False
            self.flag_ce = False
            self.flag_oe = False
            self.flag_re = False
        else:
            pass
        
    
    def ini_clock(self):
        self.time_ini = datetime.now()
        disp_time = self.time_ini.strftime('%H:%M:%S')
        self.ini_label.configure(text=disp_time)
        self.flag_ini = True
        self.flag_end = False

        ## read id participant and hide cursor
        self.id_participant=self.id_var.get()
        self.id_input.config(insertontime=0)
        print(self.id_participant)
        # checking if the directory data
        # exist or not.
        if not os.path.exists("data/"):
            # if the data directory is not present 
            # then create it.
            os.makedirs("data/")
        ## writing
        textline = f'{self.time_ini} start_recording section {self.section_var.get()}'
        self.write_data(textline)
        
        
        self.update_clock()
        
        
    def end_clock(self):
        self.time_end = datetime.now()
        self.flag_end = True
        ## writing
        textline = f'{self.time_end} end_recording section {self.section_var.get()}'
        self.write_data(textline)
        
    def ce_clock(self):
        ## restart the clock
        if self.flag_ini == True:
            self.time_ce = datetime.now()
            self.flag_ce = True
            self.flag_oe = False
            self.flag_re = False
            self.ce_label.config(bg="white")
            self.oe_label.config(bg="gray")
            self.re_label.config(bg="gray")
            
            ## writing
            textline = f'{self.time_ce} closed_eyes_start section {self.section_var.get()}'
            self.write_data(textline)
        else:
            pass

    def oe_clock(self):
        if self.flag_ini == True:
            self.time_oe = datetime.now()
            self.flag_ce = False
            self.flag_oe = True
            self.flag_re = False
            self.ce_label.config(bg="gray")
            self.oe_label.config(bg="white")
            self.re_label.config(bg="gray")
            
            ## writing
            textline = f'{self.time_oe} opened_eyes_start section {self.section_var.get()}'
            self.write_data(textline)
        else:
            pass

    def re_clock(self):
        if self.flag_ini == True:
            self.time_re = datetime.now()
            self.flag_ce = False
            self.flag_oe = False
            self.flag_re = True
            self.ce_label.config(bg="gray")
            self.oe_label.config(bg="gray")
            self.re_label.config(bg="white")
            
            ## writing
            textline = f'{self.time_re} pause_start section {self.section_var.get()}'
            self.write_data(textline)
        else:
            pass

    def display_time(self, time_ini, time_now):
        
        time_diff  = time_now - time_ini
        hh, mm, ss = self.time_conversion(time_diff)
        disp_time = str(hh).zfill(2)+':'+str(mm).zfill(2)+':'+str(ss).zfill(2)
        
        return disp_time

    def time_conversion(self, duration):
        days, seconds = duration.days, duration.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        
        return hours, minutes, seconds
        
    def print_selection(self):
        print(f'you have selected: {self.section_var.get()}')
        
    
    def write_data(self, line):
        
        filename = self.id_participant+'.dat'
        ## open the file for writing and append
        with open('data/'+filename, 'a') as f:
            f.write(f"{line}\n")
        return 0
        

app=App()

