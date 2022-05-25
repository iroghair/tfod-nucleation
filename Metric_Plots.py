import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

def metric_split(dict):
    keys = dict.keys()
    for model in keys:
        df = dict[model]
        df_prec = df[df['Value'].str.contains("Precision")]
        df_recall = df[df['Value'].str.contains("Precision")]

def precision_plot(dict):
    """CREATE PLOT SHOWING DIFFERENT PRECISION TYPES
    one subaxis per precision type, grouped by unique custom models"""
    keys = dict.keys()
    prec_fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3, sharex=True,sharey=True)
    prec_fig.suptitle('Precision Values')
    for model in keys:
        df = dict[model]
        # get different precision values from df
        # get mAP value (first row of df that contains "mAP")
        mAP = df[df['Value'].str.contains("Precision/mAP")].iloc[0].t
        mAP50IOU = df[df['Value'].str.contains("mAP@.50IOU")].t
        mAP75IOU = df[df['Value'].str.contains("mAP@.75IOU")].t
        small = df[df['Value'].str.contains("mAP") & df['Value'].str.contains("(small)")].t
        medium = df[df['Value'].str.contains("mAP") & df['Value'].str.contains("(medium)")].t
        large = df[df['Value'].str.contains("mAP") & df['Value'].str.contains("(large)")].t
        # subplot
        ax1.plot(1,mAP, marker=".",label=model)
        ax1.title.set_text('mAP')
        ax2.plot(1,mAP50IOU, marker=".",label=model)
        ax2.title.set_text('mAP @ 0.50 IoU')
        ax3.plot(1,mAP75IOU, marker=".",label=model)
        ax3.title.set_text('mAP @ 0.75 IoU')
        ax4.plot(1,small, marker=".",label=model)
        ax4.title.set_text('mAP (small)')
        ax5.plot(1,medium, marker=".",label=model)
        ax5.title.set_text('mAP (medium)')
        ax6.plot(1,large, marker=".",label=model)
        ax6.title.set_text('mAP (large)')
    #plt.legend(loc=(1.04, 0))
    plt.legend(loc='lower center', bbox_to_anchor=(-0.7, -0.6), ncol=3, fontsize = 'xx-small')
    prec_fig.subplots_adjust(bottom=0.2)
    #plt.show()
    return prec_fig

def precision_barplot(dict):
    """CREATE PLOT SHOWING DIFFERENT PRECISION TYPES
    one subaxis per precision type, grouped by unique custom models"""
    keys = dict.keys()
    prec_fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2, 3, sharex=True,sharey=True)
    #prec_fig.suptitle('Precision Values')
    x=0
    for model in keys:
        df = dict[model]
        # get different precision values from df
        # get mAP value (first row of df that contains "mAP")
        mAP = df[df['Value'].str.contains("Precision/mAP")].iloc[0].t
        mAP50IOU = df[df['Value'].str.contains("mAP@.50IOU")].t
        mAP75IOU = df[df['Value'].str.contains("mAP@.75IOU")].t
        small = df[df['Value'].str.contains("mAP") & df['Value'].str.contains("(small)")].t
        medium = df[df['Value'].str.contains("mAP") & df['Value'].str.contains("(medium)")].t
        large = df[df['Value'].str.contains("mAP") & df['Value'].str.contains("(large)")].t
        # subplot
        ax1.bar(x,mAP,width=1,label=model,align='center',alpha=.7)
        ax1.title.set_text('mAP')
        #ax1.set_ylim([0.4,1])
        ax1.set_xticks([])
        ax2.bar(x,mAP50IOU,width=1,label=model,align='center',alpha=.7)
        ax2.title.set_text('mAP @ 0.50 IoU')
        ax2.set_xticks([])
        ax3.bar(x,mAP75IOU,width=1,label=model,align='center',alpha=.7)
        ax3.title.set_text('mAP @ 0.75 IoU')
        ax3.set_xticks([])
        ax4.bar(x,small,width=1,label=model,align='center',alpha=.7)
        ax4.title.set_text('mAP (small)')
        ax4.set_xticks([])
        ax5.bar(x,medium,width=1,label=model,align='center',alpha=.7)
        ax5.title.set_text('mAP (medium)')
        ax5.set_xticks([])
        ax6.bar(x,large,width=1,label=model,align='center',alpha=.7)
        ax6.title.set_text('mAP (large)')
        ax6.set_xticks([])
        x += 1
    #plt.legend(loc=(1.04, 0))
    plt.legend(loc='lower center', bbox_to_anchor=(-0.7, -0.6), ncol=2, fontsize = 'small')
    prec_fig.subplots_adjust(bottom=0.2)
    #plt.show()
    return prec_fig

def recall_plot(dict):
    """CREATE PLOT SHOWING DIFFERENT RECALL TYPES
    one subaxis per recall type, grouped by unique custom models"""
    keys = dict.keys()
    rec_fig, ((ax1,ax2,ax3), (ax4,ax5,ax6)) = plt.subplots(2,3,sharex=True,sharey=True)
    rec_fig.suptitle('Recall Values')
    for model in keys:
        df = dict[model]
        # get different precision values from df
        # get mAP value (first row of df that contains "mAP")
        recAR1 = df[df['Value'].str.contains("AR@1")].iloc[0].t
        recAR10 = df[df['Value'].str.contains("AR@10")].iloc[0].t
        recAR100 = df[df['Value'].str.contains("AR@100")].iloc[0].t
        small = df[df['Value'].str.contains("AR@100") & df['Value'].str.contains("(small)")].t
        medium = df[df['Value'].str.contains("AR@100") & df['Value'].str.contains("(medium)")].t
        large = df[df['Value'].str.contains("AR@100") & df['Value'].str.contains("(large)")].t
        # subplot
        ax1.plot(1,recAR1, marker=".",label=model)
        ax1.title.set_text('AR @ 1')
        ax2.plot(1,recAR10, marker=".",label=model)
        ax2.title.set_text('AR @ 10')
        ax3.plot(1,recAR100, marker=".",label=model)
        ax3.title.set_text('AR @ 100')
        ax4.plot(1,small, marker=".",label=model)
        ax4.title.set_text('AR @ 100 (small)')
        ax5.plot(1,medium, marker=".",label=model)
        ax5.title.set_text('AR @ 100 (medium)')
        ax6.plot(1,large, marker=".",label=model)
        ax6.title.set_text('AR @ 100 (large)')
    #plt.legend(loc=(1.04, 0))
    plt.legend(loc='lower center', bbox_to_anchor=(-0.7, -0.6), ncol=3, fontsize = 'xx-small')
    rec_fig.subplots_adjust(bottom=0.2)
    #plt.show()
    return rec_fig

def plot_total_loss(loss_df,model_path):
    """Plot total loss (from training event file)
    tot_loss=f(time_step)"""
    plt.plot(loss_df.Step, loss_df.t)
    plt.xlabel('Step')
    plt.ylabel('Total loss')
    # get model name (last part of model path)
    model = os.path.basename(os.path.normpath(model_path))
    plt.savefig(os.path.join(model_path,f'TotLoss_{model}.png'))
    plt.close()
    print(f'Total loss plot saved in {model_path}')

def plot_learningrate(lrate_df,model_path):
    """Plot learning rate (from training event file)
    lr=f(time_step)"""
    plt.plot(lrate_df.Step, lrate_df.t)
    plt.xlabel('Step')
    plt.ylabel('Learning rate')
    # get model name (last part of model path)
    model = os.path.basename(os.path.normpath(model_path))
    plt.savefig(os.path.join(model_path,f'LearningRate_{model}.png'))
    plt.close()
    print(f'Learning rate plot saved in {model_path}')
    