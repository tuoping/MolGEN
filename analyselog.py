import os
import numpy as np
import matplotlib.pyplot as plt
import sys

font={'family':'serif',
      # 'style':'italic',  # 斜体
      'weight':'normal',
      # 'color':'red',
      'size': 18
}  
def setfigform_simple(xlabel, ylabel=None, xlimit = (None,None), ylimit = (None, None)):
    # plt.legend(fontsize = 16, frameon=False),
    plt.xlabel(xlabel, fontdict = font)
    plt.ylabel(ylabel, fontdict = font)
    plt.xlim(xlimit)
    plt.ylim(ylimit)
    # plt.xticks(fontsize = font['size'], fontname = "serif")
    # plt.yticks(fontsize = font['size'], fontname = "serif")
    plt.tick_params(direction="in")

if os.path.exists("EXCLUDE_EPOCHS"):
    exclude_epochs = np.loadtxt("EXCLUDE_EPOCHS")
else:
    exclude_epochs = []

def readlog(dir, trainlosskeyword="\'train_loss\'", ckeyword="\'train_time\'", exclude_epochs=exclude_epochs):
    alltrainsteps_baseline = []
    alltrainlosses_baseline = []
    alltraincolor_baseline = []
    with open(os.path.join(dir, "log.out")) as fp:
        lines = fp.readlines()
        for line in lines:
            l = line.split()
            if ckeyword is None:
                collect_epoch = np.array([None,None])
            else:
                collect_epoch = np.array([None,None,None])
            if trainlosskeyword in line:
                for idx_t,t in enumerate(l):
                    if "\'epoch\'" in t:
                        if float(l[idx_t+1].replace(",","")) in exclude_epochs:
                            print(float(l[idx_t+1].replace(",","")), exclude_epochs)
                            if len(alltrainlosses_baseline)>len(alltrainsteps_baseline):
                                alltrainlosses_baseline.pop(-1)
                            break
                        collect_epoch[0] = (float(l[idx_t+1].replace(",","").replace("np.float64(","").replace(")","")))
                        # break
                    if trainlosskeyword in t:
                        collect_epoch[1] = (float(l[idx_t+1].replace(",","").replace("np.float64(","").replace(")","").replace("}",'')))
                    if ckeyword is not None and ckeyword in t:
                        collect_epoch[2] = (float(l[idx_t+1].replace(",","").replace("np.float64(","").replace(")","").replace("}",'')))
                if np.any(collect_epoch is None):
                    continue
                else:
                    alltrainsteps_baseline.append(collect_epoch[0])
                    alltrainlosses_baseline.append(collect_epoch[1])
                    if ckeyword is not None:
                        alltraincolor_baseline.append(collect_epoch[2])

    if ckeyword is None:
        alltraincolor_baseline = np.arange(len(alltrainsteps_baseline))
    return np.array(alltrainlosses_baseline), np.array(alltrainsteps_baseline), np.array(alltraincolor_baseline)


def plot_1losses(dir_dir_b1024, key="\'train_loss\'", c_key=None, after_epoch=None, before_epoch=None, ymin=None, ymax=None):
    plt.rcParams["figure.figsize"] = (6,5)
    fig = plt.figure()
    alltrainlosses_dir_b1024, alltrainsteps_dir_b1024, allcolor = readlog(dir_dir_b1024, trainlosskeyword=key, ckeyword=c_key)
    alltrainlosses_dir_b1024 = np.array(alltrainlosses_dir_b1024)
    np.save(key, np.vstack([alltrainsteps_dir_b1024, alltrainlosses_dir_b1024]))
    before_idx = None
    after_idx = None
    if len(alltrainlosses_dir_b1024) != 0:
        if after_epoch is not None and after_epoch != "None":
            after_idx = np.where(np.array(alltrainsteps_dir_b1024)>=float(after_epoch))[0][0]
            print("after_idx = ", after_idx)
        if before_epoch is not None and before_epoch != "None":
            before_idx = np.where(np.array(alltrainsteps_dir_b1024)>=float(before_epoch))[0][0]
            print("before_idx = ", before_idx)
        
        
    ### remove the loss value when restart training 
    # stable_idx = np.arange(len(alltrainsteps_dir_b1024), dtype=int)
    stable_idx = np.where((np.array(alltrainsteps_dir_b1024) > 0) & (np.array(alltrainlosses_dir_b1024) < 1) )[0]
    alltrainlosses_dir_b1024 = np.array(alltrainlosses_dir_b1024)[stable_idx]
    alltrainsteps_dir_b1024 = np.array(alltrainsteps_dir_b1024)[stable_idx]
    allcolor = np.array(allcolor)[stable_idx]
    # plotting
    positive_idx = np.where(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])>0)[0][:]
    negative_idx = np.where(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])<=0)[0][:]
    print("positive_idx = ", positive_idx)
    print("negative_idx = ", negative_idx)
    print("allcolor = ", allcolor)
    print("alltrainsteps_dir_b1024 = ", alltrainsteps_dir_b1024[after_idx:before_idx])
    print("alltrainlosses_dir_b1024 = ", alltrainlosses_dir_b1024[after_idx:before_idx])
    if len(positive_idx) > len(negative_idx):
        plt.scatter(np.array(alltrainsteps_dir_b1024[after_idx:before_idx])[positive_idx], np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[positive_idx], c=allcolor[positive_idx], label="$L>0$", marker="x")
        # plt.scatter(np.arange(len(positive_idx)), np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[positive_idx], c=allcolor[positive_idx], label="$L>0$", marker="x")
        cbar = plt.colorbar()
    print("negative alltrainsteps_dir_b1024 = ", alltrainsteps_dir_b1024[after_idx:before_idx][negative_idx])
    print("negative alltrainlosses_dir_b1024 = ", alltrainlosses_dir_b1024[after_idx:before_idx][negative_idx])
    print("negative allcolor = ", allcolor[negative_idx])
    if len(positive_idx) < len(negative_idx):
        plt.scatter(np.array(alltrainsteps_dir_b1024[after_idx:before_idx])[negative_idx], np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[negative_idx], c=allcolor[negative_idx], label="$L<0$", marker="o", s=10)
        # plt.scatter(np.arange(len(negative_idx)), np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[negative_idx], c=allcolor[negative_idx], label="$L<0$", marker="o", s=10)
        cbar = plt.colorbar()
    # cbar.set_label("Ratio of conditional training per batch", fontsize=font['size']-4)
    if len(positive_idx) > 0:
        plt.axhline(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[positive_idx][-1], ls="--", c='k')
    if len(negative_idx) > 0:
        plt.axhline(np.array(alltrainlosses_dir_b1024[after_idx:before_idx])[negative_idx][-1], ls="--", c="r")
    # plt.semilogy()
    setfigform_simple("epoch",key, ylimit=(ymin, ymax))
    plt.legend()
    plt.title(dir_dir_b1024, fontdict=font)
    
    fig.tight_layout()
    plt.savefig(os.path.join(dir_dir_b1024, key), bbox_inches="tight")
    # plt.show()


import argparse

parser = argparse.ArgumentParser(description="DCD → Extended XYZ with triclinic lattice")
parser.add_argument("--after_epoch", type=int, default=None, )
parser.add_argument("--before_epoch",  type=int, default=None)
parser.add_argument("--ymin_val",  type=float, default=None)
parser.add_argument("--ymax_val",  type=float, default=None)
parser.add_argument("--ymin_train",  type=float, default=None)
parser.add_argument("--ymax_train",  type=float, default=None)
parser.add_argument("--key",  type=str, default="val_loss_gen")
args = parser.parse_args()

dir = f"./"
plot_1losses(dir, key="\'train_loss\'", after_epoch=args.after_epoch, before_epoch=args.before_epoch, ymin=args.ymin_train, ymax=args.ymax_train)
if 'train' in args.key:
    plot_1losses(dir, key=f"\'{args.key}\'", after_epoch=args.after_epoch, before_epoch=args.before_epoch, ymin=args.ymin_val, ymax=args.ymax_val)
elif 'val' in args.key:
    plot_1losses(dir, key=f"\'{args.key}\'", after_epoch=args.after_epoch, before_epoch=args.before_epoch, ymin=args.ymin_val, ymax=args.ymax_val)
else:
    plot_1losses(dir, key=f"\'{args.key}\'", after_epoch=args.after_epoch, before_epoch=args.before_epoch, ymin=args.ymin_val, ymax=args.ymax_val)
