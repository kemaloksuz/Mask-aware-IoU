import os
import pdb
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from matplotlib.ticker import StrMethodFormatter

CLS_NEG_PATH = './analysis_logs/negatives_cls_logs.npy'
CLS_POS_PATH = './analysis_logs/positives_cls_logs.npy'
POS_REG_PATH = './analysis_logs/positives_reg_logs.npy'
POS_MASK_PATH = './analysis_logs/positives_mask_logs.npy'

NEG_LOWER_LIM = 0.1
NEG_UPPER_LIM = 0.5
POS_LOWER_LIM = 0.5
POS_UPPER_LIM = 1.

def read_data(PATH, iou_lower_limit, iou_upper_limit):
    '''
    out = [[iou, anchor_rates, losses]
        ...]
    '''
    p = Path(PATH)
    with p.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = np.load(f)
        while f.tell() < fsz:
            out = np.vstack((out, np.load(f)))
    out = np.nan_to_num(out, nan=0.0)
    inds = (out[:,0]>iou_lower_limit) & (out[:,0]<iou_upper_limit)
    ious = out[inds, 0]
    anchor_rates = out[inds, 1]
    losses = out[inds, 2]

    return anchor_rates, ious, losses


def extract_stats(measure, losses, bins):
    means_ = binned_statistic(measure, losses,
                              statistic = 'mean',
                              range = (measure.min(), measure.max()),
                              bins = bins)
    
    stds  = binned_statistic(measure, losses,
                              statistic = 'std',
                              range = (measure.min(), measure.max()),
                              bins = bins)[0]

    maxs = binned_statistic(measure, losses,
                             statistic = 'max',
                             range = (measure.min(), measure.max()),
                             bins = bins)[0]

    count = binned_statistic(measure, losses,
                             statistic = 'count',
                             range = (measure.min(), measure.max()),
                             bins = bins)[0]
 
    bin_centers = means_[1]
    means = means_[0]
    bin_centers =  (bin_centers[:-1] + bin_centers[1:]) / 2
    return bin_centers, means, stds, maxs, count 


if __name__ == '__main__':
      
    # CLS LOSS PLOTS
    print("Classification Loss:")
    fig, ax = plt.subplots(1, 1)

    # plot negatives
    iou_lower_limit = NEG_LOWER_LIM
    iou_upper_limit = NEG_UPPER_LIM
    anchor_rates, ious, losses = read_data(CLS_NEG_PATH, \
                                           iou_lower_limit, iou_upper_limit)

    # try different bins
    
    bar_width = 0.04
    bar_shift = bar_width/2.0
    bin_size = 0.2#0.2, 0.125, 0.1, 0.05
    bins = np.arange(0., 1+bin_size, bin_size)
    

    bin_centers, bin_means, bin_stds, bin_max, bin_count = extract_stats(anchor_rates, losses, bins)

    bar_rate_mean = ax.bar(bin_centers-bar_shift, bin_means, width=bar_width, color='b', align='center', label='Rate Means', yerr=bin_stds, ecolor='r', linewidth=3, capsize=5.)
    
    ax_ = ax.twinx()   
    bar_rate_count = ax_.bar(bin_centers+bar_shift, bin_count, width=bar_width, color='g', align='center', label='Rate Stds.')
 
    ax.set_title('Negative Means')
    ax.set_xlabel('Anchor Rate')
    ax.set_ylabel('Cls. Loss')
    ax.set_xticks(bin_centers)
    pdb.set_trace()
    ax.set_xticklabels(('0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'))
    ax.grid(alpha=0.75, which='major', axis='y')
    ax.tick_params(axis='y', colors='b')
    ax.yaxis.label.set_color('blue')
    ax_.tick_params(axis='y', colors='g')
    ax_.yaxis.label.set_color('green')
    ax_.set_ylabel('Count')
    # equalize ylims
    ax.set_ylim(0, 1.1*(bin_means.max()+bin_stds.max()))
    ax_.set_ylim(0, 1.1*bin_count.max()) 

    title_text = 'Box CLS Loss\n'
    title_text = title_text + 'Negative: {}<IoU<{}\n'.format(iou_lower_limit, iou_upper_limit)
    
    print("----Negative Stats----\n")
    
    print("Rate_Bin_stds:{}\n".format(bin_means))
    print("Rate_Bin_var:{}\n".format(bin_stds))
    print("Rate_Bin_max:{}\n".format(bin_max))
    #plt.suptitle(title_text)
    plt.tight_layout()
    plt.show()
    #plt.savefig('./analysis_figures/loss_vs_measure/binwise/negatives_classification/BS_{}.png'.format(str(1/bin_size)))

    # plot positives 
    fig, ax = plt.subplots(1, 1)

    iou_lower_limit = POS_LOWER_LIM
    iou_upper_limit = POS_UPPER_LIM
    anchor_rates, ious, losses = read_data(CLS_POS_PATH, \
                                           iou_lower_limit, iou_upper_limit)


    bin_centers, bin_means, bin_stds, bin_max, bin_count = extract_stats(anchor_rates, losses, bins)
 
    bar_rate_mean = ax.bar(bin_centers-bar_shift, bin_means, width=bar_width, color='b', align='center', label='Anchor Rate', yerr=bin_stds, ecolor='r', capsize=3.)
    ax_ = ax.twinx() 

    bar_rate_count = ax_.bar(bin_centers+bar_shift, bin_count, width=bar_width, color='g', align='center', label='Rate Stds.')
  
    ax.set_title('Positive Means')
    ax.set_xlabel('Anchor Rate')
    ax.set_ylabel('Cls. Loss')
    ax.set_xticks(bins)
    ax.grid(alpha=0.75, which='major', axis='y')
    ax.tick_params(axis='y', colors='b')
    ax.yaxis.label.set_color('blue')
    ax_.tick_params(axis='y', colors='g')
    ax_.yaxis.label.set_color('green')
    ax.set_ylim(0, 1.1*(bin_means.max()+bin_stds.max()))
    ax_.set_ylabel('Count')

    # equalize ylims
    ax.set_ylim(0, 1.1*(bin_means.max()+bin_stds.max()))
    ax_.set_ylim(0, 1.1*bin_count.max()) 
    
    title_text = 'Box CLS Loss\n'
    title_text = title_text+'Positive: {}<IoU<{}\n'.format(iou_lower_limit, iou_upper_limit)
    
    print("----Positive Stats----\n")

    print("Rate_Bin_stds:{}\n".format(bin_means))
    print("Rate_Bin_var:{}\n".format(bin_stds))
    print("Rate_Bin_max:{}\n".format(bin_max))
    #plt.suptitle(title_text)
    plt.tight_layout()
    plt.savefig('./analysis_figures/loss_vs_measure/binwise/positives_classification/BS_{}.png'.format(str(1/bin_size)))
 
    # REG LOSS PLOTS
    print("Regression Loss:")
    
    fig, ax = plt.subplots(1, 1)


    # plot positives
    iou_lower_limit = POS_LOWER_LIM
    iou_upper_limit = POS_UPPER_LIM
    anchor_rates, ious, losses = read_data(POS_REG_PATH, \
                                           iou_lower_limit, iou_upper_limit)


    bin_centers, bin_means, bin_stds, bin_max, bin_count = extract_stats(anchor_rates, losses, bins)
 
    bar_rate_mean = ax.bar(bin_centers-bar_shift, bin_means, width=bar_width, color='b', align='center', label='Anchor Rate', yerr=bin_stds, ecolor='r', capsize=3.)        
    ax_ = ax.twinx() 
    bar_rate_count = ax_.bar(bin_centers+bar_shift, bin_count, width=bar_width, color='g', align='center', label='Rate Stds.')
  
    ax.set_title('Positive Means')
    ax.set_xlabel('Anchor Rate')
    ax.set_ylabel('Reg. Loss')
    ax.set_xticks(bins)
    ax.set_xticks(bins)
    ax.grid(alpha=0.75, which='major', axis='y')
    ax.tick_params(axis='y', colors='b')
    ax.yaxis.label.set_color('blue')
    ax_.tick_params(axis='y', colors='g')
    ax_.yaxis.label.set_color('green')
    ax.set_ylim(0, 1.1*(bin_means.max()+bin_stds.max()))
    ax_.set_ylabel('Count')

    # equalize ylims
    ax.set_ylim(0, 1.1*(bin_means.max()+bin_stds.max()))
    ax_.set_ylim(0, 1.1*bin_count.max()) 

    title_text = 'Box Reg Loss: {}<IoU<{}'.format(iou_lower_limit, iou_upper_limit)
    print("----Positives Stats----\n")

    print("Rate_Bin_stds:{}\n".format(bin_means))
    print("Rate_Bin_var:{}\n".format(bin_stds))
    print("Rate_Bin_max:{}\n".format(bin_max))
    #plt.suptitle(title_text)
    plt.tight_layout()
    plt.savefig('./analysis_figures/loss_vs_measure/binwise/positives_regression/BS_{}.png'.format(str(1/bin_size)))


    # MASK LOSS PLOTS
    print("Mask Loss:")
    
    fig, ax = plt.subplots(1, 1)


    # plot positives
    iou_lower_limit = POS_LOWER_LIM
    iou_upper_limit = POS_UPPER_LIM
    anchor_rates, ious, losses = read_data(POS_MASK_PATH, \
                                           iou_lower_limit, iou_upper_limit)


    bin_centers, bin_means, bin_stds, bin_max, bin_count = extract_stats(anchor_rates, losses, bins)
 
    bar_rate_mean = ax.bar(bin_centers-bar_shift, bin_means, width=bar_width, color='b', align='center', label='Anchor Rate', yerr=bin_stds, ecolor='r', capsize=3.)
    ax_ = ax.twinx() 
    bar_rate_count = ax_.bar(bin_centers+bar_shift, bin_count, width=bar_width, color='g', align='center', label='Anchor Rate')
  
    ax.set_title('Positive Means')
    ax.set_xlabel('Anchor Rate')
    ax.set_ylabel('Reg. Loss')
    ax.set_xticks(bins)
    ax.set_xticks(bins)
    ax.grid(alpha=0.75, which='major', axis='y')
    ax.tick_params(axis='y', colors='b')
    ax.yaxis.label.set_color('blue')
    ax_.tick_params(axis='y', colors='g')
    ax_.yaxis.label.set_color('green')
    ax.set_ylim(0, 1.1*(bin_means.max()+bin_stds.max()))
    ax_.set_ylabel('Count')

    # equalize ylims
    ax.set_ylim(0, 1.1*(bin_means.max()+bin_stds.max()))
    ax_.set_ylim(0, 1.1*bin_count.max()) 

    title_text = 'Mask Loss: {}<IoU<{}'.format(iou_lower_limit, iou_upper_limit)
    print("----Positives Stats----\n")

    print("Rate_Bin_stds:{}\n".format(bin_means))
    print("Rate_Bin_var:{}\n".format(bin_stds))
    print("Rate_Bin_max:{}\n".format(bin_max))
    #plt.suptitle(title_text)
    plt.tight_layout()
    plt.savefig('./analysis_figures/loss_vs_measure/binwise/positives_mask/BS_{}.png'.format(str(1/bin_size)))
