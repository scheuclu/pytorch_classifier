from visdom import Visdom
import numpy as np
#from configs import index2name, name2index

class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main', log_filename='/tmp'):
        self.viz = Visdom(port=6065)
        self.env = env_name
        self.plots = {}
        self.viz.log_to_filename = log_filename
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')



def plot_epoch_end(plotter, phase, epoch, epoch_acc, epoch_loss, lr, running_class_stats):
    """ Plot statistics to visdom """
    plotter.plot('acc', phase, 'acc', epoch, epoch_acc)
    plotter.plot(var_name='loss', split_name=phase, title_name='loss', x=epoch, y=epoch_loss)
    plotter.plot(var_name='LR', split_name='LR', title_name='LR', x=epoch, y=lr)
    plotter.plot(var_name='LR', split_name='LR', title_name='LR', x=epoch, y=lr)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
    # for key, val in class_acc.items():
    #     #print(key, val)
    #     classname = dataset_val.index2name[key]
    for classname in running_class_stats.keys():
        TP = running_class_stats[classname]['TP']
        TN = running_class_stats[classname]['TN']
        FP = running_class_stats[classname]['FP']
        FN = running_class_stats[classname]['FN']
        class_acc = round(float((TP + TN) / (TP + TN + FP + FN + 1e-3)), 2)
        class_prec = round(float(TP / (TP + FP + 1e-3)), 2)
        class_recall = round(float(TP / (TP + FN + 1e-3)), 2)
        print(TP, TN, FP, FN, 'ooo')
        print(class_acc, class_prec, class_recall, "---")

        # plotter_acc.plot(var_name=key, split_name=phase, title_name='class_acc', x=epoch, y=val)
        plotter.plot(var_name='acc_' + phase, split_name=classname, title_name='class_acc', x=epoch,
                     y=class_acc)
        plotter.plot(var_name='prec_' + phase, split_name=classname, title_name='class_prec', x=epoch,
                     y=class_prec)
        plotter.plot(var_name='recall_' + phase, split_name=classname, title_name='class_recall', x=epoch,
                     y=class_recall)
        plotter.plot(var_name='num_preds_' + phase, split_name=classname,
                     title_name='num_preds_', x=epoch, y=running_class_stats[classname]['num_preds'])
        plotter.plot(var_name='num_gt_' + phase, split_name=classname,
                     title_name='num_gt_', x=epoch, y=running_class_stats[classname]['num_gt'])
    return



