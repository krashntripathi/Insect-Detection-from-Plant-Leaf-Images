import os
import numpy as np
import warnings
import pandas as pd
from matplotlib import pylab
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pylab
import matplotlib.colors as mcolors

warnings.filterwarnings("ignore")

no_of_dataset = 2


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Dataset = ['Dataset 1', 'Dataset 2', 'Dataset 3']
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['Terms', 'AOOA-AMGY9-GELAN', 'FOA-AMGY9-GELAN', 'BOA-AMGY9-GELAN', 'GMEO-AMGY9-GELAN',
                 'PGMEO-AMGY9-GELAN']
    for i in range(Fitness.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((Fitness.shape[1], 5))
        for j in range(len(Algorithm) - 1):
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report', str(Dataset[i]),
              ' --------------------------------------------------')

        print(Table)
        length = np.arange(Fitness.shape[-1])
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, markersize=12, label=Algorithm[5])
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence_%s.png" % (Dataset[i]))
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence Curve of ' + str(Dataset[i]))
        plt.show()


def Plot_Confusion_():
    for n in range(no_of_dataset):
        Actual = np.load('Actual_' + str(n + 1) + '.npy', allow_pickle=True)
        Predict = np.load('Predict_' + str(n + 1) + '.npy', allow_pickle=True)
        cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1))

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display.plot(ax=ax, cmap='Blues', values_format='d', text_kw={'fontsize': 12})
        ax.set_xlabel('Predicted labels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual labels', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        if n == 0:
            Classes = [str(i).zfill(3) for i in range(103) if
                       str(i).zfill(3) not in ['031', '060', '061', '064', '076',
                                               '081', ]]  # because these labels are not have annotation
            rot = 45
            fontsizes = 5
            ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
            ax.set_yticklabels(Classes, fontsize=fontsizes)
        elif n == 1:
            Classes = ['FruitMoth', 'Gall Flies', 'Locust', 'Stem Borer']
            rot = 45
            fontsizes = 10
            ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
            ax.set_yticklabels(Classes, fontsize=fontsizes)
        else:
            Classes = ['Aphids', 'Armyworm', 'Beetle', 'Bollworm', 'Grasshopper',
                       'Mites', 'Mosquito', 'Sawfly', 'Stem borer']
            rot = 45
            fontsizes = 10
            ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
            ax.set_yticklabels(Classes, fontsize=fontsizes)
        ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
        ax.set_yticklabels(Classes, fontsize=fontsizes)
        plt.tight_layout()
        path = "./Results/Confusion_matrix_%s.png" % (n + 1)
        plt.savefig(path)
        plt.show()


def Plot_Confusion():
    Plot_Confusion_()
    results_dir = "./Results"
    os.makedirs(results_dir, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(results_dir, "Confusion_Matrices.xlsx"), engine='xlsxwriter')
    for n in range(no_of_dataset):
        Actual = np.load(f'Actual_{n + 1}.npy', allow_pickle=True)
        Predict = np.load(f'Predict_{n + 1}.npy', allow_pickle=True)
        actual_labels = np.asarray(Actual).argmax(axis=1)
        predicted_labels = np.asarray(Predict).argmax(axis=1)
        cm = confusion_matrix(actual_labels, predicted_labels)
        if n == 0:
            Classes = [str(i).zfill(3) for i in range(103) if
                       str(i).zfill(3) not in ['031', '060', '061', '064', '076',
                                               '081']]  # because these labels are not have annotation
            rot = 45
            fontsize = 5
        elif n == 1:
            Classes = ['FruitMoth', 'Gall Flies', 'Locust', 'Stem Borer']
            rot = 45
            fontsize = 10
        else:
            Classes = ['Aphids', 'Armyworm', 'Beetle', 'Bollworm', 'Grasshopper',
                       'Mites', 'Mosquito', 'Sawfly', 'Stem borer']
            rot = 45
            fontsize = 10

        df_cm = pd.DataFrame(cm, index=Classes, columns=Classes)
        df_cm.to_excel(writer, sheet_name=f'Dataset_{n + 1}')

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Classes)
        disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=rot)
        # ax.set_title(f'Confusion Matrix - Dataset {n + 1}', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Labels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Labels', fontsize=12, fontweight='bold')
        ax.set_xticklabels(Classes, fontsize=fontsize)
        ax.set_yticklabels(Classes, fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'Confusion_matrix_{n + 1}.png'))
        plt.close()
    writer.close()


def ROC_curve():
    lw = 2
    cls = ['Mask-R-CNN', 'SA-YOLOV8', 'DMSAU-Net', 'Yolov9-GELAN', 'PGMEO-AMGY9-GELAN']
    for n in range(no_of_dataset):
        Actual = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True).astype('int')
        colors = cycle(["#fe2f4a", "#0165fc", "#00ffff", "lime", "black"])
        for i, color in zip(range(len(cls)), colors):
            Predicted = np.load('Y_Score_' + str(n + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/ROC_Dataset_%s.png" % (n + 1)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('ROC curve')
        plt.savefig(path)
        plt.show()


def plot_Proposed():
    Eval = np.load('Eval_ALL.npy', allow_pickle=True)
    Terms = [
        'Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE',
        'Recall', 'Specificity', 'Precision', 'FPR',
        'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC'
    ]
    Graph_Terms = [0, 1, 2, 3, 5]
    colors = ['#ff006e', '#fca311', '#99582a', '#b56576', 'k']
    Images = ['1', '2', '3', '4', '5']
    stats = np.zeros((no_of_dataset, len(Graph_Terms), 5, 5))
    for p in range(no_of_dataset):
        Eval_all = Eval[p]
        for i, metric_idx in enumerate(Graph_Terms):
            for r in range(5):
                for j in range(5):
                    values = Eval_all[r, j][:, metric_idx + 4]
                    stats[p, i, j, 0] = np.max(values)
                    stats[p, i, j, 1] = np.min(values)
                    stats[p, i, j, 2] = np.mean(values)
                    stats[p, i, j, 3] = np.median(values)
                    stats[p, i, j, 4] = np.std(values)

        for i, metric_idx in enumerate(Graph_Terms):
            Graph = stats[p, i, :, 2]
            fig = plt.figure(figsize=(9, 6), facecolor='#dcf8e3')
            ax = fig.add_axes([0.1, 0.15, 0.8, 0.75])
            ax.set_facecolor('#dcf8e3')
            X = np.arange(len(Graph))
            barWidth = 0.5
            ax.bar(X, Graph, width=barWidth, color=colors, edgecolor='#032b43')
            for xi, height in zip(X, Graph):
                ax.text(
                    xi,
                    height + height * 0.02,
                    f"{np.round(height, 3)}",
                    ha='center',
                    va='bottom',
                    fontsize=11,
                    fontweight='bold'
                )

            ax.set_xticks(X)
            ax.set_xticklabels(Images, fontsize=11, fontweight='bold')
            plt.xlabel('Images', fontsize=12, fontweight='bold', color='#5f0f40')
            plt.ylabel(Terms[metric_idx], fontsize=12, fontweight='bold', color='#5f0f40')
            plt.yticks(fontsize=12, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            save_path = f'./Results/Dataset_{p + 1}_Proposed_{Terms[metric_idx]}.png'
            plt.savefig(save_path)
            plt.show()
            plt.close()


def plot_mAP():
    eval = np.load('Eval_ALL_mAP.npy', allow_pickle=True)
    Terms = ['mAP']
    Graph_Terms = [0]
    Algorithms = ['AOOA-AMGY9-GELAN', 'FOA-AMGY9-GELAN', 'BOA-AMGY9-GELAN', 'GMEO-AMGY9-GELAN', 'PGMEO-AMGY9-GELAN']
    Classifiers = ['Mask-R-CNN', 'SA-YOLOV8', 'DMSAU-Net', 'Yolov9-GELAN', 'PGMEO-AMGY9-GELAN']
    ACT = ['4', '8', '16', '32']
    inner_colors = ['#E57373', '#81C784', '#7986CB', '#BA68C8', '#9E9E9E']
    outer_colors = ['#C62828', '#2E7D32', '#283593', '#6A1B9A', '#212121']
    light_colors = [mcolors.to_rgba(c, alpha=0.3) for c in inner_colors]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :, :, Graph_Terms[j]]
            Algorithm_values = Graph[:4, :5].transpose()

            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 1.25
            group_gap = 1.4
            index = np.arange(len(ACT)) * (bar_width * Algorithm_values.shape[0] + group_gap)
            ax.yaxis.grid(True, linestyle='-', alpha=0.6)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=12))
            for i in range(Algorithm_values.shape[0]):
                ax.bar(
                    index + i * bar_width,
                    Algorithm_values[i],
                    bar_width,
                    color=light_colors[i % len(light_colors)],
                    edgecolor=outer_colors[i % len(outer_colors)],
                    linewidth=2,
                    label=Algorithms[i]
                )

            ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_ylabel(Terms[Graph_Terms[j]] + ' →', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_xticks(index + bar_width * (Algorithm_values.shape[0] / 2 - 0.5))
            ax.set_xticklabels(ACT)
            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=outer_colors[i], markersize=12)
                for i in range(Algorithm_values.shape[0])]

            ax.legend(legend_handles, Algorithms, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False,
                      prop={'weight': 'bold'})

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.savefig(f"./Results/BS_{Terms[Graph_Terms[j]]}_Dataset_{n + 1}_Alg.png")
            # plt.show()
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            Method_values = Graph[:4, 5:].transpose()
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 1.25
            group_gap = 1.4
            index = np.arange(len(ACT)) * (bar_width * Method_values.shape[0] + group_gap)
            ax.yaxis.grid(True, linestyle='-', alpha=0.6)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=12))
            for i in range(Method_values.shape[0]):
                ax.bar(
                    index + i * bar_width,
                    Method_values[i],
                    bar_width,
                    color=light_colors[i % len(light_colors)],
                    edgecolor=outer_colors[i % len(outer_colors)],
                    linewidth=2,
                    label=Classifiers[i]
                )

            ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_ylabel(Terms[Graph_Terms[j]] + ' →', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_xticks(index + bar_width * (Method_values.shape[0] / 2 - 0.5))
            ax.set_xticklabels(ACT)
            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=outer_colors[i], markersize=12)
                for i in range(Method_values.shape[0])]

            ax.legend(legend_handles, Classifiers, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3,
                      frameon=False,
                      prop={'weight': 'bold'})

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.savefig(f"./Results/BS_{Terms[Graph_Terms[j]]}_Dataset_{n + 1}_mtd.png")
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def plot_BS():
    Eval = np.load('Eval_ALL_BS.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC']
    Graph_terms = [0, 1, 2, 3, 5, 7]
    Algorithms = ['AOOA-AMGY9-GELAN', 'FOA-AMGY9-GELAN', 'BOA-AMGY9-GELAN', 'GMEO-AMGY9-GELAN', 'PGMEO-AMGY9-GELAN']
    Classifiers = ['Mask-R-CNN', 'SA-YOLOV8', 'DMSAU-Net', 'Yolov9-GELAN', 'PGMEO-AMGY9-GELAN']
    ACT = ['4', '8', '16', '32']
    stats = np.zeros((len(Graph_terms), Eval.shape[-3] + 1, 5))
    inner_colors = ['#E57373', '#81C784', '#7986CB', '#BA68C8', '#9E9E9E']
    outer_colors = ['#C62828', '#2E7D32', '#283593', '#6A1B9A', '#212121']
    light_colors = [mcolors.to_rgba(c, alpha=0.3) for c in inner_colors]
    for n in range(no_of_dataset):
        Eval_all = Eval[n]
        for k, g_idx in enumerate(Graph_terms):
            for r in range(5):
                for j in range(Eval_all.shape[-3]):
                    data = Eval_all[r, j][:, g_idx + 4]
                    stats[k, j, 0] = np.max(data)
                    stats[k, j, 1] = np.min(data)
                    stats[k, j, 2] = np.mean(data)
                    stats[k, j, 3] = np.median(data)
                    stats[k, j, 4] = np.std(data)

            stats[k, 9, :] = stats[k, 4, :]
            Alg_Val = stats[k, :5, :-1]
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 1.25
            group_gap = 1.4
            index = np.arange(len(ACT)) * (bar_width * Alg_Val.shape[0] + group_gap)
            ax.yaxis.grid(True, linestyle='-', alpha=0.6)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=12))
            for i in range(Alg_Val.shape[0]):
                ax.bar(
                    index + i * bar_width,
                    Alg_Val[i],
                    bar_width,
                    color=light_colors[i % len(light_colors)],
                    edgecolor=outer_colors[i % len(outer_colors)],
                    linewidth=2,
                    label=Algorithms[i]
                )

            ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_ylabel(Terms[g_idx] + ' →', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_xticks(index + bar_width * (Alg_Val.shape[0] / 2 - 0.5))
            ax.set_xticklabels(ACT)
            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=outer_colors[i], markersize=12)
                for i in range(Alg_Val.shape[0])]

            ax.legend(legend_handles, Algorithms, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False,
                      prop={'weight': 'bold'})

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.savefig(f"./Results/BS_{Terms[g_idx]}_Dataset_{n + 1}_Alg.png")
            plt.show(block=False)
            plt.pause(1)
            plt.close()

            #  CLASSIFIER PLOT
            Mtd_Val = stats[k, 5:, :-1]
            fig, ax = plt.subplots(figsize=(10, 6))
            index = np.arange(len(ACT)) * (bar_width * Mtd_Val.shape[0] + group_gap)
            ax.yaxis.grid(True, linestyle='-', alpha=0.6)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=12))
            num_cls = len(Classifiers)
            Mtd_Val = stats[k, 5:5 + num_cls, :-1]
            for i in range(num_cls):
                ax.bar(
                    index + i * bar_width,
                    Mtd_Val[i],
                    bar_width,
                    color=light_colors[i % len(light_colors)],
                    edgecolor=outer_colors[i % len(outer_colors)],
                    linewidth=2,
                    label=Classifiers[i]
                )
            ax.set_xlabel('Batch Size', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_ylabel(Terms[g_idx] + ' →', fontsize=12, fontweight='bold', color='#35530a')
            ax.set_xticks(index + bar_width * (Mtd_Val.shape[0] / 2 - 0.5))
            ax.set_xticklabels(ACT)
            legend_handles = [
                plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=outer_colors[i], markersize=12)
                for i in range(Mtd_Val.shape[0])
            ]
            ax.legend(
                legend_handles,
                Classifiers,
                loc='upper center',
                bbox_to_anchor=(0.5, 1.15),
                ncol=3,
                frameon=False,
                prop={'weight': 'bold'}
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.savefig(f"./Results/BS_{Terms[g_idx]}_Dataset_{n + 1}_Mtd.png")
            plt.show(block=False)
            plt.pause(1)
            plt.close()


def plot_Epoch():
    Eval = np.load('Eval_all_EP.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_terms = [0, 1, 2, 3, 5, 13]

    Full = ['Epoch', 'AOOA-AMGY9-GELAN', 'FOA-AMGY9-GELAN', 'BOA-AMGY9-GELAN', 'GMEO-AMGY9-GELAN', 'PGMEO-AMGY9-GELAN',
            'Mask-R-CNN', 'SA-YOLOV8', 'DMSAU-Net', 'Yolov9-GELAN', 'PGMEO-AMGY9-GELAN']

    Losses = ['Mse loss', 'Categorical loss', 'Binary loss', 'Center loss', 'SparseCategoricalCrossentropy class']
    ACT = ['20', '40', '60', '80', '100', '120']
    Table_stats = np.zeros((Eval.shape[-4], len(Graph_terms), Eval.shape[-3], 5))
    for n in range(no_of_dataset):
        for p in range(Eval.shape[-4]):
            Eval_all = Eval[n][p]
            for i in range(len(Graph_terms)):
                for j in range(Eval_all.shape[-3]):
                    if j < Eval_all.shape[-3]:
                        Table_stats[p, i, j, 0] = np.max(Eval_all[j][:, Graph_terms[i] + 4])
                        Table_stats[p, i, j, 1] = np.min(Eval_all[j][:, Graph_terms[i] + 4])
                        Table_stats[p, i, j, 2] = np.mean(Eval_all[j][:, Graph_terms[i] + 4])
                        Table_stats[p, i, j, 3] = np.median(Eval_all[j][:, Graph_terms[i] + 4])
                        Table_stats[p, i, j, 4] = np.std(Eval_all[j][:, Graph_terms[i] + 4])

        alg_prop = Table_stats[:, :, 4, :]
        Table_stats[:, :, 9, :] = alg_prop
        for t in range(len(Graph_terms)):
            Table = PrettyTable()
            Table.add_column(Full[0], ACT)
            for k in range(5):  # len(Full) - 1
                Table.add_column(Full[k + 1], Table_stats[:, t, k, 2])
            print(
                '-------------------------------------------------- Mean Algorithm Comparison of Dataset ' + str(n + 1),
                Terms[Graph_terms[t]],
                '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Full[0], ACT)
            for k in range(5, 10):
                Table.add_column(Full[k + 1], Table_stats[:, t, k, 2])
            print('-------------------------------------------------- Mean Classifier Comparison of Dataset ',
                  str(n + 1), Terms[Graph_terms[t]],
                  '--------------------------------------------------')
            print(Table)


def plot_Abliation():
    eval = np.load('Eval_all_Ablation.npy', allow_pickle=True)
    Terms = ['Accuracy']
    # Classifier = ['TERMS', 'DNN', 'NN', 'Dilated NN', 'Atrous NN', 'Convolutional NN', 'Residual NN', 'Multiscale NN',
    # 'Trans NN', 'Multicross atrous NN', 'WDNN']
    Classifier = ['TERMS', 'Capsule Network', 'CNN', 'Dilated Capsule Network', 'Dilated Convolutional Neural Network',
                  'LSTM', 'Mtd_1', 'Mtd_2', 'Mtd_3', 'Mtd_4', 'PROPOSED']
    for n in range(eval.shape[0]):
        Value = eval[n, 4, :, :]
        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(5, len(Classifier) - 1):
            formatted_values = [f"{value:.2f}" for value in Value[j, :]]
            Table.add_column(Classifier[j + 1], formatted_values)
        print('-------------------------------------------------- Abliation Experiment of Dataset ' + str(n + 1),
              '--------------------------------------------------')
        print(Table)


def Image_Results():
    import numpy as np
    import cv2 as cv
    import matplotlib.pyplot as plt

    # Dataset 1 label mapping
    dataset1_labels = {
        '001': "Rice Leaf Roller",
        '002': "Rice Leaf Caterpillar",
        '003': "Paddy Stem Maggot",
        '004': "Asiatic Rice Borer",
        '005': "Yellow Rice Borer",
        '006': "Rice Gall Midge",
        '007': "Rice Stemfly",
        '008': "Brown Plant Hopper",
        '009': "White Backed Plant Hopper",
        '010': "Small Brown Plant Hopper",
        '011': "Rice Water Weevil",
        '012': "Rice Leafhopper",
        '013': "Grain Spreader Thrips",
        '014': "Rice Shell Pest",
        '015': "Grub",
        '016': "Mole Cricket",
        '017': "Wireworm",
        '018': "White Margined Moth",
        '019': "Black Cutworm",
        '020': "Large Cutworm",
        '021': "Yellow Cutworm",
        '022': "Red Spider",
        '023': "Corn Borer",
        '024': "Army Worm",
        '025': "Aphids",
        '026': "Potosiabre Vitarsis",
        '027': "Peach Borer",
        '028': "English Grain Aphid",
        '029': "Green Bug",
        '030': "Bird Cherry-Oataphid",
        '032': "Penthaleus Major",
        '033': "Longlegged Spider Mite",
        '034': "Wheat Phloeothrips",
        '035': "Wheat Sawfly",
        '036': "Cerodonta Denticornis",
        '037': "Beet Fly",
        '038': "Flea Beetle",
        '039': "Cabbage Army Worm",
        '040': "Beet Army Worm",
        '041': "Beet Spot Flies",
        '042': "Meadow Moth",
        '043': "Beet Weevil",
        '044': "Sericaorient Alismots Chulsky",
        '045': "Alfalfa Weevil",
        '046': "Flax Budworm",
        '047': "Alfalfa Plant Bug",
        '048': "Tarnished Plant Bug",
        '049': "Locustoidea",
        '050': "Lytta Polita",
        '051': "Legume Blister Beetle",
        '052': "Blister Beetle",
        '053': "Therioaphis Maculata Buckton",
        '054': "Odontothrips loti",
        '055': "Thrips",
        '056': "Alfalfa Seed Chalcid",
        '057': "Pieris Canidia",
        '058': "Apolygus Lucorum",
        '059': "Limacodidae",
        '062': "Brevipoalpus Lewisi McGregor",
        '063': "Oides Decempunctata",
        '065': "Pseudococcus Comstocki Kuwana",
        '066': "Parathrene Regalis",
        '067': "Ampelophaga",
        '068': "Lycorma Delicatula",
        '069': "Xylotrechus",
        '070': "Cicadella Viridis",
        '071': "Miridae",
        '072': "Trialeurodes Vaporariorum",
        '073': "Erythroneura Apicalis",
        '074': "Papilio Xuthus",
        '075': "Panonchus Citri McGregor",
        '077': "Icerya Purchasi Maskell",
        '078': "Unaspis Yanonensis",
        '079': "Ceroplastes Rubens",
        '080': "Chrysomphalus Aonidum",
        '082': "Nipaecoccus Vastalor",
        '083': "Aleurocanthus Spiniferus",
        '084': "Tetradacus C Bactrocera Minax",
        '085': "Dacus Dorsalis(Hendel)",
        '086': "Bactrocera Tsuneonis",
        '087': "Prodenia Litura",
        '088': "Adristyrannus",
        '089': "Phyllocnistis Citrella Stainton",
        '090': "Toxoptera Citricidus",
        '091': "Toxoptera Aurantii",
        '092': "Aphis Citricola Vander Goot",
        '093': "Scirtothrips Dorsalis Hood",
        '094': "Dasineura Sp",
        '095': "Lawana Imitata Melichar",
        '096': "Salurnis Marginella Guerr",
        '097': "Deporaus Marginatus Pascoe",
        '098': "Chlumetia Transversa",
        '099': "Mango Flat Beak Leafhopper",
        '100': "Rhytidodera Bowrinii White",
        '101': "Sternochetus Frigidus",
        '102': "Cicadellidae"
    }

    # Indexes for images to show for each dataset
    index_1 = [6580, 10348, 12273]
    index_2 = [2913, 2936, 2984]
    index = [index_1, index_2]

    for n in range(no_of_dataset):
        Images = np.load(f'Images_{n + 1}.npy', allow_pickle=True)
        Segmented = np.load(f'Detected_{n + 1}.npy', allow_pickle=True)
        Target = np.load(f'Target_{n + 1}.npy', allow_pickle=True)

        for i in range(len(index[n])):
            img_idx = index[n][i]

            image = cv.resize(Images[img_idx], (512, 512))
            seg = cv.resize(Segmented[img_idx], (512, 512))

            # Determine label for this image
            target_entry = Target[img_idx]

            # Determine label for this image
            target_entry = Target[img_idx]

            if n == 0:
                # Dataset 1: use dataset1_labels
                if isinstance(target_entry, str) or isinstance(target_entry, np.str_):
                    label = target_entry
                else:
                    if hasattr(target_entry, 'shape') and len(target_entry.shape) > 0:
                        class_idx = np.argmax(target_entry)
                    else:
                        class_idx = int(target_entry)
                    key = str(class_idx).zfill(3)
                    label = dataset1_labels.get(key, "Unknown")

            elif n == 1:
                # Dataset 2: use this specific list
                labels = ['FruitMoth', 'Gall Flies', 'Locust', 'Stem Borer']
                if isinstance(target_entry, str) or isinstance(target_entry, np.str_):
                    label = target_entry
                else:
                    if hasattr(target_entry, 'shape') and len(target_entry.shape) > 0:
                        class_idx = np.argmax(target_entry)
                    else:
                        class_idx = int(target_entry)
                    label = labels[class_idx] if class_idx < len(labels) else "Unknown"

            # Draw label background for visibility
            cv.rectangle(seg, (5, 5), (400, 50), (0, 0, 0), -1)

            # Put label text on images
            cv.putText(seg, label, (10, 35), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

            # Convert BGR to RGB for matplotlib
            image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            seg_rgb = cv.cvtColor(seg, cv.COLOR_BGR2RGB)

            plt.figure(figsize=(10, 5))
            plt.suptitle(f"Image {i + 1} from dataset {n + 1}", fontsize=20)

            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(image_rgb)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.title('Detected Image')
            plt.imshow(seg_rgb)
            plt.axis('off')

            # Save results
            path1 = f"./Results/Image_Results/Dataset_{n + 1}_image_{i + 1}.png"
            plt.savefig(path1)
            # plt.show()
            np.save('plot_original.npy', image)
            cv.imwrite(f'./Results/Image_Results/Orig_{i + 1}_Dataset_{n + 1}.png', image)
            cv.imwrite(f'./Results/Image_Results/Pest_{i + 1}_Dataset_{n + 1}.png', seg)


def Sample_images():
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        if Target.shape[-1] >= 2:
            targ = np.argmax(Target, axis=1).reshape(-1, 1)
        else:
            targ = Target
        class_indices = {}
        for class_label in np.unique(targ):
            indices = np.where(targ == class_label)[0]
            class_indices[class_label] = indices
        for class_label, indices in class_indices.items():
            if n == 0:
                labels = [str(i).zfill(3) for i in range(103) if
                          str(i).zfill(3) not in ['031', '060', '061', '064', '076',
                                                  '081']]  # because these labels are not have annotation
            elif n == 1:
                labels = ['FruitMoth', 'Gall Flies', 'Locust', 'Stem Borer']
            else:
                labels = ['Aphids', 'Armyworm', 'Beetle', 'Bollworm', 'Grasshopper',
                          'Mites', 'Mosquito', 'Sawfly', 'Stem borer']
            if len(indices) >= 5:
                no_samples = 5
            else:
                no_samples = len(indices)
            for i in range(no_samples):
                print(n, no_of_dataset, labels[class_label], i)
                Image = cv.resize(Images[indices[i]], (512, 512))
                cv.imshow('Image', Image)
                cv.waitKey(750)
                cv.imwrite('./Results/Sample_Images/Dataset_' + str(n + 1) + '_' + str(
                    labels[class_label]) + '_image_' + str(i + 1) + '.png', Image)


def Explainable_AI():
    for n in range(no_of_dataset):
        # Load PET and XAI images
        PET_Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        XAI_Images = np.load('XAI_images_' + str(n + 1) + '.npy', allow_pickle=True)

        # All_Image indices
        All_Image = [
            [190, 111, 205, 207, 362],
            [275, 357, 921, 1007, 1409]]

        # Loop over the images in the current dataset (n)
        for i in range(len(All_Image[n])):
            print(f"Processing image {i + 1} of 5 in Dataset {n + 1}")

            # Ensure proper indexing by using All_Image[n][i]
            pet_img = cv.resize(PET_Images[All_Image[n][i]], (512, 512))
            xai_img = cv.resize(XAI_Images[All_Image[n][i]], (512, 512))

            # Display images if needed
            cv.imshow('PET Image', pet_img)
            cv.imshow('XAI Image', xai_img)
            cv.waitKey(0)

            # Save the XAI image with appropriate filename
            cv.imwrite('./Results/XAI/Dataset_' + str(n + 1) + '_Gradcam_Image_%04d.png' % (i + 1), xai_img)
            cv.imwrite('./Results/XAI/Dataset_' + str(n + 1) + '_Original_Image_%04d.png' % (i + 1), pet_img)

        print(f"Dataset {n + 1} processing complete.")


if __name__ == '__main__':
    plot_conv()
    ROC_curve()
    Plot_Confusion_()
    Plot_Confusion()
    plot_Proposed()
    plot_BS()
    plot_Epoch()
    plot_mAP()
    plot_Abliation()
    Image_Results()
    Sample_images()
    Explainable_AI()
