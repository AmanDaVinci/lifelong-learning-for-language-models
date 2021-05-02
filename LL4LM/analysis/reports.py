import pandas as pd
from LL4LM.analysis.processing import get_experiment_data
from LL4LM.analysis.metrics import measure_lifelong_metrics
from LL4LM.analysis.plotting import (
    plot_lifelong_curve, 
    plot_gradient_interference, 
    plot_gradient_overlap
)


def generate_full_report(run_id, multitask_run_id, unitask_run_id, save_path=None):
    exp_logs, stream, name = get_experiment_data(run_id)
    mtl_logs, _, _ = get_experiment_data(multitask_run_id)
    utl_logs, _, _ = get_experiment_data(unitask_run_id)
    df = measure_lifelong_metrics("Lifelong", stream, exp_logs, mtl_logs, utl_logs)
    training_fig = plot_lifelong_curve(
        name, stream, exp_logs, mtl_logs, utl_logs,
        training=True, testing=False, testing_detailed=False
    )
    testing_fig = plot_lifelong_curve(
        name, stream, exp_logs, mtl_logs, utl_logs,
        training=False, testing=True, testing_detailed=False
    )
    testing_detailed_fig = plot_lifelong_curve(
        name, stream, exp_logs, mtl_logs, utl_logs,
        training=False, testing=True, testing_detailed=True
    )
    interference_fig = plot_gradient_interference("Lifelong", stream, exp_logs, mtl_logs)
    overlap_fig = plot_gradient_overlap("Lifelong", stream, exp_logs, mtl_logs)
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        training_fig.savefig(save_path/"lifelong_training.png")
        testing_fig.savefig(save_path/"lifelong_testing.png")
        testing_detailed_fig.savefig(save_path/"lifelong_testing_detailed.png")
        interference_fig.savefig(save_path/"lifelong_gradient_interference.png")
        overlap_fig.savefig(save_path/"lifelong_gradient_overlap.png")
        df.to_csv(save_path/"lifelong_metrics.csv")
    return df, training_fig, testing_fig, testing_detailed_fig, interference_fig, overlap_fig

def generate_comparative_report(run_ids, multitask_run_id, unitask_run_id):
    mtl_logs, _, _ = get_experiment_data(multitask_run_id)
    utl_logs, _, _ = get_experiment_data(unitask_run_id)
    report = []
    for run_id in run_ids:
        exp_logs, stream, name = get_experiment_data(run_id)
        df = measure_lifelong_metrics("Lifelong", stream, exp_logs, mtl_logs, utl_logs)
        report.append(df.loc["Average"].rename(name))
    return pd.concat(report, axis=1).T