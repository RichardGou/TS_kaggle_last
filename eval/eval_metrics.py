import torch
import numpy as np
import ruptures as rpt
from ruptures.metrics import hausdorff


def synthetic_hausdorff_distances(ts_0, true, bkp_1):
    ts_0 = ts_0.cpu().numpy()
    true = true.cpu().numpy()
    bkp_1 = bkp_1.cpu().numpy()
    ts_concat = np.concatenate((ts_0, true), axis=1)
    batch_size, sequence_length, _ = ts_concat.shape
    hausdorff_distances = []

    for i in range(batch_size):
        signal = ts_concat[i, :, 0]  
        true_bkp = np.array([bkp_1[i],sequence_length]) 
        algo = rpt.Dynp(model="l2").fit(signal)
        predicted_bkp = np.array(algo.predict(n_bkps=1))
        hausdorff_dist = hausdorff(true_bkp, predicted_bkp)
        hausdorff_distances.append(hausdorff_dist)

    return np.array(hausdorff_distances).mean()

def detect_anomalies(ts, threshold=3):
    device = ts.device  
    mean = torch.mean(ts)
    std = torch.std(ts)
    anomalies = torch.where((ts > mean + threshold * std) | (ts < mean - threshold * std))[0].to(device)
    last_index = torch.tensor([len(ts) - 1], dtype=torch.long, device=device)
    first_index = torch.tensor([0], dtype=torch.long, device=device)
    if anomalies.numel() == 0 or anomalies[0] != first_index:
        anomalies = torch.cat((first_index.to(device), anomalies))
    if anomalies.numel() == 0 or anomalies[-1] != last_index:
        anomalies = torch.cat((anomalies, last_index.to(device)))
    return anomalies

def traffic_hausdorff_distances(ts_0, true, preds):
    hausdorff_distances = []
    true_ts = torch.cat((ts_0, true), axis=1)
    pred_ts = torch.cat((ts_0, preds), axis=1)
    batch_size, _, _ = true_ts.shape
    for i in range(batch_size):
        anomalies_true = detect_anomalies(true_ts[i])
        anomalies_preds =  detect_anomalies(pred_ts[i])
        hausdorff_dist = hausdorff(anomalies_true.cpu().numpy(), anomalies_preds.cpu().numpy())
        hausdorff_distances.append(hausdorff_dist)
    return np.array(hausdorff_distances).mean()


def swinging_doors_batch(data_batch, epsilon):
    batch_size, seq_len, _ = data_batch.size()
    data_batch = data_batch.squeeze(-1)  
    SD_ts = torch.zeros_like(data_batch)
    
    for b in range(batch_size):
        data = data_batch[b]
        lower_slope = float('inf')
        upper_slope = float('-inf')
        start_idx = 0
        
        for i in range(1, seq_len):
            lower_slope = min(lower_slope, (data[i] - (data[start_idx] - epsilon)) / (i - start_idx))
            upper_slope = max(upper_slope, (data[i] - (data[start_idx] + epsilon)) / (i - start_idx))
            
            if lower_slope < upper_slope or i == seq_len - 1:
                current_slope = (data[i] - data[start_idx]) / (i - start_idx)
                SD_ts[b, start_idx:i] = current_slope
                start_idx = i
                lower_slope = float('inf')
                upper_slope = float('-inf')
        
        SD_ts[b, -1] = current_slope
    
    return SD_ts.unsqueeze(-1)


def ramp_score_batch(true_batch, predicted_batch, epsilon):
    ramp_approx_1 = swinging_doors_batch(true_batch, epsilon)
    ramp_approx_2 = swinging_doors_batch(predicted_batch, epsilon)
    return torch.mean(torch.abs(ramp_approx_1-ramp_approx_2))