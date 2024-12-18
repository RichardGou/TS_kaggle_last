import matplotlib.pyplot as plt
import os
import torch

def plot_preds(X_train, y_train, preds, save_path="figures", file_name="plot.png"):
    os.makedirs(save_path, exist_ok=True)
    
    save_file = os.path.join(save_path, file_name)

    plt.figure(figsize=(12, 6))

    plt.plot(range(len(X_train)), X_train, label="Past values", color="blue")
    plt.plot(range(len(X_train), len(X_train) + len(y_train)), y_train, label="Actual values", color="red")

    for name, predictions in preds.items():
        plt.plot(
            range(len(X_train), 
                  len(X_train) + len(predictions)), 
                  predictions, 
                  label=f"Predicted values ({name})", 
                  linestyle="--",
                  )
        
    plt.axvline(x=len(X_train), color="black", linestyle=":", label="Start of predictions")
    plt.title("Time Series Example (Train, Actual, Predictions)")
    plt.xlabel("Time (indices)")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)

    plt.savefig(save_file, bbox_inches="tight")
    print(f"Figure saved to {save_file}")

    plt.show()


def plot_all(net_gru_dilate, net_gru_mse, net_gru_dtw, net_gru_rrmse, testloader, data):
    gen_test = iter(testloader)
    batches_to_process = 2 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_gru_dilate = net_gru_dilate.to(device)
    net_gru_mse = net_gru_mse.to(device)
    net_gru_dtw = net_gru_dtw.to(device)
    net_gru_rrmse = net_gru_rrmse.to(device)

    for k in range(batches_to_process):
        if data=="traffic":
            test_inputs, test_targets = next(gen_test)
        elif data=="power":
            test_inputs, test_targets = next(gen_test)
        else:
            test_inputs, test_targets, _ = next(gen_test)

        test_inputs = test_inputs.to(torch.float32).to(device)
        test_targets = test_targets.to(torch.float32).to(device)

        batch_size = test_inputs.size(0)
        random_indices = torch.randint(0, batch_size, (1,)) 

        preds = {"MSE": [], "DILATE": [], "sDTW": [], "RRMSE": []}

        with torch.no_grad():
            preds_mse_batch = net_gru_mse(test_inputs).squeeze(-1).detach().cpu().numpy()
            preds_dilate_batch = net_gru_dilate(test_inputs).squeeze(-1).detach().cpu().numpy()
            preds_dtw_batch = net_gru_dtw(test_inputs).squeeze(-1).detach().cpu().numpy()
            preds_rrmse_batch = net_gru_rrmse(test_inputs).squeeze(-1).detach().cpu().numpy()
            i=0
            for ind in random_indices:
                preds["MSE"].append(preds_mse_batch[ind])
                preds["DILATE"].append(preds_dilate_batch[ind])
                preds["sDTW"].append(preds_dtw_batch[ind])
                preds["RRMSE"].append(preds_rrmse_batch[ind])
                X_true = test_inputs[ind].squeeze(-1).cpu().numpy()
                y_true = test_targets[ind].squeeze(-1).cpu().numpy()
                plot_preds(
                    X_true, 
                    y_true, 
                    {"MSE": preds["MSE"][-1], "DILATE": preds["DILATE"][-1], "sDTW": preds["sDTW"][-1], "RRMSE": preds["RRMSE"][-1]},
                    save_path="figures/predictions", 
                    file_name=f"time_series_plot_{(i,k)}_{data}.png"
                )
                i+=1


def plot_metrics_gammas(metrics, data):
    gammas = metrics["gamma"]

    plt.figure(figsize=(10, 6))
    for metric_name in ["mse", "dtw", "tdi", "hausdorff", "ramp"]:
        plt.plot(
            gammas, metrics[metric_name],
            label=metric_name, marker='o'
        )

    plt.xscale('log') 
    plt.xlabel("Gamma (log scale)")
    plt.ylabel("Metric value")
    plt.title("Evaluation Metrics for Different Gammas")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    output_dir = "figures/gammas"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"metrics_plot_gammas_{data}.png")
    plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def plot_metrics_vs_alpha(metrics, data):
    alphas = metrics["alpha"]

    plt.figure(figsize=(12, 8))
    for metric_name in ["mse", "dtw", "tdi", "hausdorff", "ramp"]:
        plt.plot(
            alphas, metrics[metric_name],
            label=metric_name, marker='o'
        )

    plt.xlabel("Alpha")
    plt.ylabel("Metric Value")
    plt.title("Metrics vs Alpha")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5)

    output_dir = "figures/alphas"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"metrics_plot_alphas_{data}.png")
    plt.savefig(output_path, bbox_inches="tight")

    plt.show()