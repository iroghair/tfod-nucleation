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
        ax1.plot(1,mAP, marker="o",label=model)
        ax1.title.set_text('mAP')
        ax2.plot(1,mAP50IOU, marker="o",label=model)
        ax2.title.set_text('mAP @ 0.50 IoU')
        ax3.plot(1,mAP75IOU, marker="o",label=model)
        ax3.title.set_text('mAP @ 0.75 IoU')
        ax4.plot(1,small, marker="o",label=model)
        ax4.title.set_text('mAP (small)')
        ax5.plot(1,medium, marker="o",label=model)
        ax5.title.set_text('mAP (medium)')
        ax6.plot(1,large, marker="o",label=model)
        ax6.title.set_text('mAP (large)')
    #plt.legend(loc=(1.04, 0))
    plt.legend(loc='lower center', bbox_to_anchor=(-0.7, -0.5), ncol=3)
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
        ax1.plot(1,recAR1, marker="o",label=model)
        ax1.title.set_text('AR @ 1')
        ax2.plot(1,recAR10, marker="o",label=model)
        ax2.title.set_text('AR @ 10')
        ax3.plot(1,recAR100, marker="o",label=model)
        ax3.title.set_text('AR @ 100')
        ax4.plot(1,small, marker="o",label=model)
        ax4.title.set_text('AR @ 100 (small)')
        ax5.plot(1,medium, marker="o",label=model)
        ax5.title.set_text('AR @ 100 (medium)')
        ax6.plot(1,large, marker="o",label=model)
        ax6.title.set_text('AR @ 100 (large)')
    #plt.legend(loc=(1.04, 0))
    plt.legend(loc='lower center', bbox_to_anchor=(-0.7, -0.5), ncol=3)
    rec_fig.subplots_adjust(bottom=0.2)
    #plt.show()
    return rec_fig
