import numpy as np
import matplotlib.pyplot as plt
from LL4LM.analysis.processing import get_train_accuracies, get_test_accuracies


plt.style.use("seaborn-talk")

def plot_lifelong_curve(name, stream, logs, multitask_logs, unitask_logs, 
                        training=True, testing=False, testing_detailed=False):
    fig, ax = plt.subplots(figsize=(12,6), constrained_layout=True, dpi=300)

    boundaries = np.cumsum([dataset_examples for _, dataset_examples in stream])
    for boundary in boundaries:
        ax.vlines(x=boundary, ymin=0, ymax=1, linestyle="dashed", color="gray")
    ax.set_xticks(boundaries)
    ax.set_xticklabels(boundaries, rotation="vertical")
    top_xaxis = ax.secondary_xaxis("top")
    top_xaxis.set_xticks(boundaries)
    top_xaxis.set_xticklabels([name for name, _ in stream], rotation="vertical")

    if training:
        exp_accuracies = get_train_accuracies(logs, rolling_window=20)
        mtl_accuracies = get_train_accuracies(multitask_logs, rolling_window=20)
        ax.plot(exp_accuracies.index, exp_accuracies.values, 
                label=f"{name} Training", color="tab:orange")
        ax.plot(mtl_accuracies.index, mtl_accuracies.values, 
                 label="Multi-task Training", color="tab:pink", alpha=0.5)
        for dataset_name, _ in stream:
            utl_accuracy = get_train_accuracies(unitask_logs, 20, dataset_name)
            ax.plot(utl_accuracy.index, utl_accuracy.values, 
                    color="blue", alpha=0.3, label="Uni-task Training")
    elif testing:
        exp_accuracies = get_test_accuracies(logs, stream)
        mtl_accuracies = get_test_accuracies(multitask_logs, stream)
        utl_accuracies = get_test_accuracies(unitask_logs, stream, unitask=True)
        if testing_detailed:
            dataset_colors = plt.cm.Set1(range(len(stream)))
            for (dataset_name, _), boundary, color in zip(stream, boundaries, dataset_colors):
                ax.plot(exp_accuracies.index, exp_accuracies[dataset_name], color=color, alpha=0.5)
                ax.vlines(x=boundary, ymin=0, ymax=1, linestyle="dashed", color=color)
                ax.scatter(x=boundary, y=utl_accuracies[dataset_name], color=color, marker="o")
        else:
            ax.plot(exp_accuracies.index, exp_accuracies.mean(axis=1), 
                    label=f"{name} Testing", color="tab:orange")
            ax.plot(mtl_accuracies.index, mtl_accuracies.mean(axis=1), 
                    label="Multi-task Testing", color="tab:pink", alpha=0.5)
            for (dataset_name, _), boundary in zip(stream, boundaries):
                ax.scatter(x=boundary, y=utl_accuracies[dataset_name], 
                           label="Uni-task Testing", color="blue", alpha=0.5, marker="o")
        
    ax.set_xlabel("Streaming Examples")
    ax.set_ylabel("Accuracy")
    ax.grid(False, axis='x')
    ax.grid(True, axis='y')
    if not testing_detailed:
        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:3], labels[:3], loc="lower left")
    return fig
