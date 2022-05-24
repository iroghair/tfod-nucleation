from TFODPaths import get_paths_and_files
from SeparateTest_Functions import get_time_diff_name
import os
import numpy as np
import pandas as pd
import ast
import seaborn as sns
from matplotlib import pyplot as plt
from ODE_bubble_growth import get_Rb

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
    exp1 = df[df.prop=='bubble_number_Exp1']
    exp2 = df[df.prop=='bubble_number_Exp2']
    exp3 = df[df.prop=='bubble_number_Exp3']
    exp4 = df[df.prop=='bubble_number_Exp4']
    plt.plot(exp3["time [min]"], exp3.num, 'o',color="tab:blue",markersize=2,label=(r'$\zeta$'+"=0.05"))
    plt.plot(exp1["time [min]"], exp1.num, 'o', color="tab:orange", markersize=2,label=(r'$\zeta$'+"=0.09"))
    plt.plot(exp4["time [min]"], exp4.num, 'o',color="tab:red",markersize=2,label=(r'$\zeta$'+"=0.14"))
    plt.plot(exp2["time [min]"], exp2.num, 'o',color="tab:purple",markersize=2,label=(r'$\zeta$'+"=0.21"))
    plt.xlabel('Time [min]')
    plt.ylabel('Bubble count [-]')
    #plt.title("Total number of bubbles")
    plt.legend()
    plt.savefig(os.path.join(save_path,f'BubbleCount_AllExp.png'),bbox_inches="tight")
    plt.close()
    print(f'Bubble count plot saved under {save_path}')

def plot_area(df, save_path):
    # get timestamps in min
    exp1 = df[df.Exp=='1']
    exp2 = df[df.Exp=='2']
    exp3 = df[df.Exp=='3']
    exp4 = df[df.Exp=='4']
    plt.plot(exp3["time [min]"], exp3["area [mm^2]"], 'o',color="tab:blue",markersize=2,label=(r'$\zeta$'+"=0.05"))
    plt.plot(exp1["time [min]"], exp1["area [mm^2]"], 'o', color="tab:orange", markersize=2,label=(r'$\zeta$'+"=0.09"))
    plt.plot(exp4["time [min]"], exp4["area [mm^2]"], 'o',color="tab:red",markersize=2,label=(r'$\zeta$'+"=0.14"))
    plt.plot(exp2["time [min]"], exp2["area [mm^2]"], 'o',color="tab:purple",markersize=2,label=(r'$\zeta$'+"=0.21"))
    plt.xlabel('Time [min]')
    plt.ylabel('Total bubble surface area [mm^{2}]')
    plt.legend()
    plt.savefig(os.path.join(save_path,f'BubbleArea_AllExp.png'),bbox_inches="tight")
    plt.close()
    print(f'Bubble area plot saved under {save_path}')

def plot_avrg_Bdiam(df, save_path):
    """Plot average bubble diameter over time
    (timestamps must be indicated in image name)
    input: df containing average bubble diameter"""
    exp1 = df[df.prop=='avrg_diam_Exp1']
    exp2 = df[df.prop=='avrg_diam_Exp2']
    exp3 = df[df.prop=='avrg_diam_Exp3']
    exp4 = df[df.prop=='avrg_diam_Exp4']
    # ODE numerical solution
    R_b1, t1 = get_Rb(p_0=6,p_S=5.5,t_max=round(max(exp1["time [min]"])*60)) # bubble radius [m], time [s]
    R_b2, t2 = get_Rb(p_0=5.7,p_S=4.7,t_max=round(max(exp2["time [min]"])*60)) # bubble radius [m], time [s]
    R_b3, t3 = get_Rb(p_0=5.8,p_S=5.5,t_max=round(max(exp3["time [min]"])*60)) # bubble radius [m], time [s]
    R_b4, t4 = get_Rb(p_0=5.4,p_S=4.7,t_max=round(max(exp4["time [min]"])*60)) # bubble radius [m], time [s]
    t_ODE1 = t1 / 60 # time vector [min]
    t_ODE2 = t2 / 60 # time vector [min]
    t_ODE3 = t3 / 60 # time vector [min]
    t_ODE4 = t4 / 60 # time vector [min]
    d_B_ODE_m1 = R_b1 * 2 * 10**(3) # bubble diameter [mm]
    d_B_ODE_m2 = R_b2 * 2 * 10**(3) # bubble diameter [mm]
    d_B_ODE_m3 = R_b3 * 2 * 10**(3) # bubble diameter [mm]
    d_B_ODE_m4 = R_b4 * 2 * 10**(3) # bubble diameter [mm]
    plt.plot(t_ODE3,d_B_ODE_m3, color="lightskyblue",label=(r'$\zeta$'+"=0.05 (Model)"))
    plt.plot(t_ODE1,d_B_ODE_m1, color="moccasin",label=(r'$\zeta$'+"=0.09 (Model)"))
    plt.plot(exp3["time [min]"], exp3.diam, 'o',color="tab:blue",markersize=2,label=(r'$\zeta$'+"=0.05"))
    plt.plot(exp1["time [min]"], exp1.diam, 'o',color="tab:orange",markersize=2,label=(r'$\zeta$'+"=0.09"))
    plt.plot(t_ODE4,d_B_ODE_m4, color="salmon",label=(r'$\zeta$'+"=0.14 (Model)"))
    plt.plot(t_ODE2,d_B_ODE_m2, color="plum",label=(r'$\zeta$'+"=0.21 (Model)"))
    plt.plot(exp4["time [min]"], exp4.diam, 'o',color="tab:red",markersize=2,label=(r'$\zeta$'+"=0.14"))
    plt.plot(exp2["time [min]"], exp2.diam, 'o',color="tab:purple",markersize=2,label=(r'$\zeta$'+"=0.21"))
    plt.xlabel('Time [min]')
    plt.ylabel('$D_b$(t) [mm]')#Average bubble diameter
    ax = plt.gca()
    ax.set_ylim([0, 1.5])
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=4)
    plt.savefig(os.path.join(save_path,f'Avrg_Diam_AllExp.png'),bbox_inches="tight")
    plt.close()
    print(f'Average diameter plot saved under {save_path}')

def plot_avrg_Bdiam_sqrt(df, save_path):
    """Plot average bubble diameter over time
    (timestamps must be indicated in image name)
    input: df containing average bubble diameter"""
    exp1 = df[df.prop=='avrg_diam_Exp1']
    exp2 = df[df.prop=='avrg_diam_Exp2']
    exp3 = df[df.prop=='avrg_diam_Exp3']
    exp4 = df[df.prop=='avrg_diam_Exp4']
    # ODE numerical solution
    R_b1, t1 = get_Rb(p_0=6,p_S=5.5,t_max=round(max(exp1["time [min]"])*60)) # bubble radius [m], time [s]
    R_b2, t2 = get_Rb(p_0=5.7,p_S=4.7,t_max=round(max(exp2["time [min]"])*60)) # bubble radius [m], time [s]
    R_b3, t3 = get_Rb(p_0=5.8,p_S=5.5,t_max=round(max(exp3["time [min]"])*60)) # bubble radius [m], time [s]
    R_b4, t4 = get_Rb(p_0=5.4,p_S=4.7,t_max=round(max(exp4["time [min]"])*60)) # bubble radius [m], time [s]
    t_ODE1 = t1 / 60 # time vector [min]
    t_ODE2 = t2 / 60 # time vector [min]
    t_ODE3 = t3 / 60 # time vector [min]
    t_ODE4 = t4 / 60 # time vector [min]
    d_B_ODE_m1 = R_b1 * 2 * 10**(3) # bubble diameter [mm]
    d_B_ODE_m2 = R_b2 * 2 * 10**(3) # bubble diameter [mm]
    d_B_ODE_m3 = R_b3 * 2 * 10**(3) # bubble diameter [mm]
    d_B_ODE_m4 = R_b4 * 2 * 10**(3) # bubble diameter [mm]
    #R_b, t = get_Rb() # bubble radius [m], time [s]
    #t_ODE = t / 60 # time vector [min]
    #d_B_ODE_m = R_b * 2 # bubble diameter [m]
    #d_B_ODE = d_B_ODE_m * 10**(3) # bubble diameter [mm]
    plt.plot(np.sqrt(t_ODE3),d_B_ODE_m3, color="lightskyblue",label=(r'$\zeta$'+"=0.05 (Model)"))
    plt.plot(np.sqrt(t_ODE1),d_B_ODE_m1, color="moccasin",label=(r'$\zeta$'+"=0.09 (Model)")) #, label="Epstein, Plesset model")
    plt.plot(np.sqrt(exp3["time [min]"]), exp3.diam, 'o',color="tab:blue",markersize=2,label=(r'$\zeta$'+"=0.05"))
    plt.plot(np.sqrt(exp1["time [min]"]), exp1.diam, 'o',color="tab:orange",markersize=2,label=(r'$\zeta$'+"=0.09"))
    plt.plot(np.sqrt(t_ODE4),d_B_ODE_m4, color="salmon",label=(r'$\zeta$'+"=0.14 (Model)"))
    plt.plot(np.sqrt(t_ODE2),d_B_ODE_m2, color="plum",label=(r'$\zeta$'+"=0.21 (Model)"))
    plt.plot(np.sqrt(exp4["time [min]"]), exp4.diam, 'o',color="tab:red",markersize=2,label=(r'$\zeta$'+"=0.14"))
    plt.plot(np.sqrt(exp2["time [min]"]), exp2.diam, 'o',color="tab:purple",markersize=2,label=(r'$\zeta$'+"=0.21"))
    plt.xlabel('\u221At')
    plt.ylabel('$D_b$(t) [mm]')#Average bubble diameter
    #plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4)#, fontsize = 'xx-small') #loc='upper left'
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.35), ncol=4)#, fontsize = 'xx-small') #loc='upper left'
    ax = plt.gca()
    ax.set_ylim([0, 1.5])
    plt.savefig(os.path.join(save_path,f'Avrg_Diam_sqrt_AllExp.png'),bbox_inches="tight")
    plt.close()
    print(f'Average diameter plot (sqrt) saved under {save_path}')

def plot_avrg_unmatch_Bdiam(df, save_path):
    """Plot average bubble diameter over time of unmatched bboxes
    (timestamps must be indicated in image name)
    input: df containing average bubble diameter"""
    exp1 = df[df.prop=='diam_unmatched_Exp1']
    exp2 = df[df.prop=='diam_unmatched_Exp2']
    exp3 = df[df.prop=='diam_unmatched_Exp3']
    exp4 = df[df.prop=='diam_unmatched_Exp4']
    plt.plot(exp3["time [min]"], exp3.diam, 'o',color="tab:blue",markersize=2,label=(r'$\zeta$'+"=0.05"))
    plt.plot(exp1["time [min]"], exp1.diam, 'o',color="tab:orange",markersize=2,label=(r'$\zeta$'+"=0.09"))
    plt.plot(exp4["time [min]"], exp4.diam, 'o',color="tab:red",markersize=2,label=(r'$\zeta$'+"=0.14"))
    plt.plot(exp2["time [min]"], exp2.diam, 'o',color="tab:purple",markersize=2,label=(r'$\zeta$'+"=0.21"))
    plt.xlabel('Time [min]')
    plt.ylabel('$D_{b, Tracked}$(t) [mm]') #Average bubble diameter
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=2)
    plt.savefig(os.path.join(save_path,f'Diam_Unmatched_AllExp.png'),bbox_inches="tight")
    plt.close()
    print(f'Average diameter (unmatched) plot saved under {save_path}')

def plot_accum_num(df, save_path):
    """Plot accumulated number of tracked/unmatched bubbles"""
    exp1 = df[df.prop=="n_unmatched_Exp1"]
    exp2 = df[df.prop=="n_unmatched_Exp2"]
    exp3 = df[df.prop=="n_unmatched_Exp3"]
    exp4 = df[df.prop=="n_unmatched_Exp4"]
    plt.plot(exp3["time [min]"], exp3["Accum Num"], 'o',color="tab:blue",markersize=2,label=(r'$\zeta$'+"=0.05"))
    plt.plot(exp1["time [min]"], exp1["Accum Num"], 'o', color="tab:orange", markersize=2,label=(r'$\zeta$'+"=0.09"))
    plt.plot(exp4["time [min]"], exp4["Accum Num"], 'o',color="tab:red",markersize=2,label=(r'$\zeta$'+"=0.14"))
    plt.plot(exp2["time [min]"], exp2["Accum Num"], 'o',color="tab:purple",markersize=2,label=(r'$\zeta$'+"=0.21"))
    plt.xlabel('Time [min]')
    plt.ylabel('Accumulated number of tracked bubbles [-]')
    plt.legend()
    plt.savefig(os.path.join(save_path,f'AccNum_AllExp.png'),bbox_inches="tight")
    plt.close()
    print(f'Bubble area plot saved under {save_path}')


##########################################################
# indicate custom model & desired checkpoint from training
CUSTOM_MODEL = 'my_centernet_hg104_1024_8'

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
        if "DSC" in lines[0][0]:
            time_names = [line[0] for line in lines]
            splits = [t.split('__')[1] for t in time_names]
            abs_times = pd.to_datetime(splits,format='%H%M%S') #.to_series()
            start_time = min(abs_times)
            # time differences (timedelta)
            rel_times = [t - start_time for t in abs_times]
            rel_times_tot = [t.total_seconds() for t in rel_times]
            rel_times_tot_str = [str(t)+' s' for t in rel_times_tot]
            # concatenate time names and property from lines
            lines = [[t,line[1]] for t,line in zip(rel_times_tot_str,lines)]
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
# add column including number of experiment
bub_num_df['Exp'] = bub_num_df['prop'].str.split('Exp').str[1]
avrg_diam_df['Exp'] = avrg_diam_df['prop'].str.split('Exp').str[1]
# calculate bubble area
area_df = pd.merge(bub_num_df,avrg_diam_df,on=["time [s]", "Exp"])
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
plot_avrg_unmatch_Bdiam(unmatch_diam_df, data_path)

# accumulated number of unmatched bubbles
accum_unmatch_df = unmatch_num_df.copy()
accum_unmatch_df["Accum Num"] = np.nan
for p in accum_unmatch_df.prop.unique():
    unmatch_num = accum_unmatch_df.num[accum_unmatch_df.prop==p]
    accum_unmatch_df.loc[accum_unmatch_df.prop==p,"Accum Num"] = np.cumsum(unmatch_num)
plot_accum_num(accum_unmatch_df, data_path)

x=1



