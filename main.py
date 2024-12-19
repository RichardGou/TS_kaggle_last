import warnings
import warnings; warnings.simplefilter('ignore')
import argparse
import os
import json

import torch
import random

from data.synthetic_dataset.synthetic_dataset import get_synthetic_data
from data.traffic.traffic_dataset import get_traffic_data
from data.power_consumption.power_dataset import get_electricity_data
from models.train_eval import compare_models, compare_gammas, compare_alphas
from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from visualization.visu import plot_all, plot_metrics_gammas, plot_metrics_vs_alpha


def opts() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default="synthetic", help="Dataset to use: 'synthetic', 'traffic' or 'power'")
    parser.add_argument('--path_data', type=str, default=None, help="Path to dataset to use: 'traffic' or 'power'")
    parser.add_argument('--stride_traffic', type=int, default="20")
    parser.add_argument('--n_epochs', type=int, default=3, help="Number of epochs")
    parser.add_argument('--logger', action='store_true', help="Enable logging to wandb")
    parser.add_argument('--no-logger', dest='logger', action='store_false', help="Disable logging to wandb")
    parser.add_argument('--gamma', type=str, default="0.01")
    parser.add_argument('--alpha', type=str, default="0.5")
    parser.add_argument('--train', action='store_true', help="Enable training")
    parser.add_argument('--no-train', dest='train', action='store_false', help="Disable training")
    parser.add_argument('--viz', action='store_true', help="Enable visualization")
    parser.add_argument('--no-viz', dest='viz', action='store_false', help="Disable visualization")
    parser.add_argument('--eval_every', type=int, default="5")
    
    args = parser.parse_args()
    args = parser.parse_args()
    return args


def main():
    args = opts()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    random.seed(0)
    n_epochs = args.n_epochs
    wandb_logger = args.logger

    if args.data == "synthetic":
        output_length = 20
        batch_size = 100
        trainloader, validloader, testloader = get_synthetic_data(output_length, batch_size=batch_size)

    elif args.data == "traffic":
        output_length = 36
        print("OUTPUT LENGTH TRAFFIC : ",output_length)
        path_data = args.path_data or "data/traffic/traffic.txt"
        batch_size=32
        stride = args.stride_traffic
        trainloader, validloader, testloader = get_traffic_data(path_data=path_data, stride=stride, 
                                                                batch_size=batch_size, output_length=output_length)
        
    elif args.data =="power":
        path_data = args.path_data or "data/power_consumption/powerconsumption.csv"
        output_length = 24
        batch_size = 16
        trainloader, validloader, testloader = get_electricity_data(path_data=path_data, batch_size=batch_size, output_length=output_length)

    else:
        raise ValueError("Invalid dataset")


    if args.gamma == "choice":

        gammas = [0.0001,0.001, 0.01, 0.1, 1]
        metrics_gammas = compare_gammas(
            gammas=gammas,
            output_length=output_length, 
            device=device, 
            batch_size=batch_size, 
            trainloader=trainloader, 
            validloader=validloader, 
            testloader=testloader, 
            n_epochs=n_epochs,
            wandb_logger=wandb_logger,
            data=args.data,
            eval_every=args.eval_every
            )
        
        repertoire = "metrics"
        fichier_json = os.path.join(repertoire, f"gammas_{args.data}.json")
        os.makedirs(repertoire, exist_ok=True)
        with open(fichier_json, "w") as f:
            json.dump(metrics_gammas, f, indent=4)

        if args.viz:
            plot_metrics_gammas(metrics_gammas, data=args.data)

    elif args.gamma.replace('.', '', 1).isdigit() and args.alpha == "choice":

        gamma = float(args.gamma)
        alphas = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        metrics_alphas = compare_alphas(
            alphas=alphas,
            gamma=gamma,
            output_length=output_length, 
            device=device, 
            batch_size=batch_size, 
            trainloader=trainloader, 
            validloader=validloader, 
            testloader=testloader, 
            n_epochs=n_epochs,
            wandb_logger=wandb_logger,
            data=args.data,
            eval_every=args.eval_every
            )

        repertoire = f"metrics"
        fichier_json = os.path.join(repertoire, f"alphas_{args.data}.json")
        os.makedirs(repertoire, exist_ok=True)
        with open(fichier_json, "w") as f:
            json.dump(metrics_alphas, f, indent=4)

        if args.viz:
            plot_metrics_vs_alpha(metrics_alphas, data=args.data)


    elif args.gamma.replace('.', '', 1).isdigit() and args.alpha.replace('.', '', 1).isdigit():
        gamma = float(args.gamma)
        alpha = float(args.alpha)

        encoder_dilate = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
        decoder_dilate = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
        net_gru_dilate = Net_GRU(encoder_dilate,decoder_dilate, output_length, device).to(device)

        encoder_mse = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
        decoder_mse = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
        net_gru_mse = Net_GRU(encoder_mse,decoder_mse, output_length, device).to(device)

        encoder_dtw = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
        decoder_dtw = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
        net_gru_dtw = Net_GRU(encoder_dtw,decoder_dtw, output_length, device).to(device)

        encoder_rrmse = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
        decoder_rrmse = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1,fc_units=16, output_size=1).to(device)
        net_gru_rrmse = Net_GRU(encoder_rrmse,decoder_rrmse, output_length, device).to(device)

        compare_models(
            training=args.train, 
            net_gru_dilate=net_gru_dilate, 
            net_gru_mse=net_gru_mse, 
            net_gru_dtw=net_gru_dtw, 
            net_gru_rrmse=net_gru_rrmse,
            trainloader=trainloader, 
            validloader=validloader, 
            testloader=testloader, 
            device=device, 
            n_epochs=n_epochs, 
            gamma=gamma, 
            alpha=alpha,
            wandb_logger=wandb_logger,
            data=args.data,
            eval_every=args.eval_every,
        )

        if args.viz:
            plot_all(
                net_gru_dilate=net_gru_dilate, 
                net_gru_mse=net_gru_mse, 
                net_gru_dtw=net_gru_dtw, 
                net_gru_rrmse=net_gru_rrmse,
                testloader=testloader,
                data=args.data)

    else:
        raise ValueError(f"Invalid value for gamma: {args.gamma}")


if __name__ == "__main__":
    main()
