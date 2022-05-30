from TFODPaths import get_paths_and_files
from DetectAnalysis_Functions import get_time_diff_name
import os
import numpy as np
import pandas as pd
import ast
import seaborn as sns
from matplotlib import pyplot as plt
from ODE_bubble_growth import get_Rb

"""Kalman Filter based Centroid Tracker - plot functions"""

size=15
params = {'legend.fontsize': 'large',
          #'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'font.size': size*0.65,
          'axes.titlepad': 25}
plt.rcParams.update(params)

def plot_Bcount(df, save_path):
    """Plot number of detected bubbles over time
    (timestamps must be indicated in image name)
    input: df containing number of detected bubbles"""
    # get timestamps in min
    plt.plot(df["time [min]"], df.num, 'o',color="tab:blue",markersize=2,label=(r'$\zeta$'+"=0.16"))
    plt.xlabel('Time [min]')
    plt.ylabel('Bubble count [-]')
    #plt.title("Total number of bubbles")
    plt.legend()
    plt.savefig(os.path.join(save_path,f'BubbleCount_016.png'),bbox_inches="tight")
    plt.close()
    print(f'Bubble count plot saved under {save_path}')

def plot_area(df, save_path):
    # get timestamps in min
    plt.plot(df["time [min]"], df["area [mm^2]"], 'o',color="tab:blue",markersize=2,label=(r'$\zeta$'+"=0.16"))
    plt.xlabel('Time [min]')
    plt.ylabel('Total bubble surface area [mm^2]')
    plt.legend()
    plt.savefig(os.path.join(save_path,f'BubbleArea_016.png'),bbox_inches="tight")
    plt.close()
    print(f'Bubble area plot saved under {save_path}')

def plot_avrg_Bdiam(df, save_path):
    """Plot average bubble diameter over time
    (timestamps must be indicated in image name)
    input: df containing average bubble diameter"""
    # ODE numerical solution
    R_b, t = get_Rb() # bubble radius [m], time [s]
    t_ODE = t / 60 # time vector [min]
    d_B_ODE_m = R_b * 2 # bubble diameter [m]
    d_B_ODE = d_B_ODE_m * 10**(3) # bubble diameter [mm]
    plt.plot(df["time [min]"], df.diam, 'o',color="tab:blue",markersize=2,label=(r'$\zeta$'+"=0.16"))
    plt.plot(t_ODE,d_B_ODE, color="tab:green", label="Epstein, Plesset model")
    plt.xlabel('Time [min]')
    plt.ylabel('$D_b$(t) [mm]')#Average bubble diameter
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    plt.savefig(os.path.join(save_path,f'Avrg_Diam_016.png'),bbox_inches="tight")
    plt.close()
    print(f'Average diameter plot saved under {save_path}')

def plot_avrg_Bdiam_sqrt(df, save_path):
    """Plot average bubble diameter over time
    (timestamps must be indicated in image name)
    input: df containing average bubble diameter"""
    # ODE numerical solution
    R_b, t = get_Rb() # bubble radius [m], time [s]
    t_ODE = t / 60 # time vector [min]
    d_B_ODE_m = R_b * 2 # bubble diameter [m]
    d_B_ODE = d_B_ODE_m * 10**(3) # bubble diameter [mm]
    plt.plot(np.sqrt(t_ODE),d_B_ODE,color="tab:green", label="Epstein, Plesset model")
    plt.plot(np.sqrt(df["time [min]"]), df.diam, 'o',color="tab:blue",markersize=4,label=(r'$\zeta$'+"=0.16"))
    plt.xlabel('\u221At')
    plt.ylabel('$D_b$(t) [mm]')#Average bubble diameter
    plt.legend()#, fontsize = 'xx-small') #loc='upper left'
    plt.savefig(os.path.join(save_path,f'Avrg_Diam_sqrt_016.png'),bbox_inches="tight")
    plt.close()
    print(f'Average diameter plot (sqrt) saved under {save_path}')

def plot_avrg_unmatch_Bdiam(df_unmatch, df_detec, save_path):
    """Plot average bubble diameter over time of unmatched bboxes
    (timestamps must be indicated in image name)
    input: df containing average bubble diameter"""
    plt.plot(df_unmatch["time [min]"], df_unmatch.diam, 'o',color="tab:orange",markersize=5,label=(r'$\zeta$'+"=0.16 (Unmatched)"))
    plt.plot(df_detec["time [min]"], df_detec.diam, 'o',color="tab:green",markersize=5,label=(r'$\zeta$'+"=0.16 (Detected)"))
    plt.xlabel('Time [min]')
    plt.ylabel('$D_{b}$(t) [mm]') #Average bubble diameter
    plt.legend()
    plt.savefig(os.path.join(save_path,f'Diam_Unmatched_016.png'),bbox_inches="tight")
    plt.close()
    print(f'Average diameter (unmatched) plot saved under {save_path}')

def plot_accum_num(df, save_path):
    """Plot accumulated number of tracked/unmatched bubbles"""
    plt.plot(df["time [min]"], df["Accum Num"], 'o',color="tab:blue",markersize=5,label=(r'$\zeta$'+"=0.16"))
    plt.xlabel('Time [min]')
    plt.ylabel('$n_{Acc}$ tracked bubbles [-]')
    plt.legend()
    plt.savefig(os.path.join(save_path,f'AccNum_016.png'),bbox_inches="tight")
    plt.close()
    print(f'Bubble area plot saved under {save_path}')


##########################################################
# indicate custom model & desired checkpoint from training
CUSTOM_MODEL = 'my_centernet_hg104_1024_7'

# get paths and files of custom model
paths, files = get_paths_and_files(CUSTOM_MODEL)
data_path=os.path.join(paths['IMAGE_PATH'], 'tested',CUSTOM_MODEL)

file_paths = []
for file in os.listdir(data_path):
    if file.endswith(".txt"):
        file_paths.append(os.path.join(data_path, file))

bub_nums = []
avrg_diams = []
unmatch_nums = []
unmatch_diams = []
for filename in file_paths:
    # read txt file
    with open(filename) as file:
        # read line by line and separate elements of row
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
        # convert str representation of list to list
        lines = [ast.literal_eval(line) for line in lines]
        # if img names instead of time names, change to time names
        if "t" in lines[0][0]:
            time_names = [line[0] for line in lines]
            splits = [t.split('_')[0] for t in time_names]
            times = [t.split('t')[1] for t in splits]
            times_s = [float(t)*60 for t in times]
            times_str = [str(t)+' s' for t in times_s]
            # concatenate time names and property from lines
            lines = [[t,line[1]] for t,line in zip(times_str,lines)]
        # remove unit [s]
        lines = [[line[0].split("s")[0],line[1]] for line in lines]
        # add property name
        prop = os.path.basename(os.path.normpath(filename)).split(".txt")[0]
        for line in lines:
            line.append(prop)
        if 'bubble_number' in prop:
            bub_nums = bub_nums + lines
        elif 'avrg_diam' in prop:
            avrg_diams = avrg_diams + lines
        elif 'n_unmatched' in prop:
            unmatch_nums = unmatch_nums + lines
        elif 'diam_unmatched' in prop:
            unmatch_diams = unmatch_diams + lines

# get average of unmatched bubble diameters
unmatch_diams_avrg = [[line[0],sum(line[1])/len(line[1]),line[2]] for line in unmatch_diams]
# put lists into dataframes
bub_num_df = pd.DataFrame(bub_nums, columns=["time [s]","num","prop"])
avrg_diam_df = pd.DataFrame(avrg_diams, columns=["time [s]", "diam", "prop"])
# bboxes of bubbles that could not be matched in subsequent img
unmatch_num_df = pd.DataFrame(unmatch_nums, columns=["time [s]", "num", "prop"])
unmatch_diam_df = pd.DataFrame(unmatch_diams_avrg, columns=["time [s]", "diam", "prop"])
# calculate bubble area
area_df = pd.merge(bub_num_df,avrg_diam_df,on=["time [s]"])
area_df["area [mm^2]"] = area_df.num * 4 * np.pi * (area_df.diam/2)**2

# convert time [s] to [min]
def time_sec_to_min(df):
    times = round(df["time [s]"].astype(float)) / 60
    df["time [min]"] = times
time_sec_to_min(bub_num_df)
time_sec_to_min(avrg_diam_df)
time_sec_to_min(area_df)
time_sec_to_min(unmatch_num_df)
time_sec_to_min(unmatch_diam_df)

# plots
plot_Bcount(bub_num_df, data_path)
plot_avrg_Bdiam(avrg_diam_df, data_path)
plot_avrg_Bdiam_sqrt(avrg_diam_df, data_path)
plot_avrg_unmatch_Bdiam(unmatch_diam_df,avrg_diam_df, data_path)

# accumulated number of unmatched bubbles
accum_unmatch_df = unmatch_num_df.copy()
accum_unmatch_df["Accum Num"] = np.nan
for p in accum_unmatch_df.prop.unique():
    unmatch_num = accum_unmatch_df.num[accum_unmatch_df.prop==p]
    accum_unmatch_df.loc[accum_unmatch_df.prop==p,"Accum Num"] = np.cumsum(unmatch_num)
plot_accum_num(accum_unmatch_df, data_path)