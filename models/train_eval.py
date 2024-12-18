import torch
import numpy as np
import wandb

from models.seq2seq import EncoderRNN, DecoderRNN, Net_GRU
from loss.dilate_loss import dilate_loss
from tslearn.metrics import dtw_path
from eval.eval_metrics import ramp_score_batch, synthetic_hausdorff_distances, traffic_hausdorff_distances

def train_model(net, loss_type, learning_rate, trainloader, validloader, device, data_type, epochs=1000, gamma = 0.001,
                print_every=50, verbose=1, alpha=0.5, wandb_logger=False, early_stopping=True, patience=15):
    import copy
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()

    best_eval_loss = float('inf')  # On cherche à minimiser la perte
    current_loss = None
    current_best_model = None
    epochs_no_improve = 0
    stop_training = False

    if wandb_logger:
        if loss_type == "dilate" and alpha >= 1-1e-5:
            name_loss = "sDTW"
        else:
            name_loss = loss_type
        logger = wandb.init(
            project="DILATE",
            name=f"model_{data_type}_{gamma}_{alpha}_{name_loss}",  
        )
    
    for epoch in range(epochs): 
        # Entraînement
        for i, data in enumerate(trainloader, 0):
            inputs = data[0]
            target = data[1]
            inputs = inputs.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)                    

            outputs = net(inputs)
            loss_mse, loss_shape, loss_temporal = torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(0, device=device)
            
            if loss_type == 'mse':
                loss_mse = criterion(target, outputs)
                loss = loss_mse

            elif loss_type == 'rrmse':
                mse_loss = torch.nn.functional.mse_loss(outputs, target, reduction='mean')
                target_mean_squared = torch.mean(target**2)
                loss_rrmse = torch.sqrt(mse_loss / target_mean_squared)
                loss = loss_rrmse

            elif loss_type == 'dilate':
                loss, loss_shape, loss_temporal = dilate_loss(target, outputs, alpha, gamma, device)             

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if wandb_logger:
            logger.log({
                'train_loss': loss.item(),
                'loss_shape': loss_shape.item(),
                'loss_temporal': loss_temporal.item(),
            })          
      
        if verbose and (epoch % print_every == 0):
            print('epoch ', epoch, ' loss ', loss.item(), ' loss shape ', loss_shape.item(), ' loss temporal ', loss_temporal.item())
            final_mse, final_dtw, final_tdi, final_hausdorff, final_ramp = eval_model(net, validloader, 
                                                                                      device=device, data_type=data_type)
            print('Eval mse= ', final_mse,
                  ' dtw= ', final_dtw,
                  ' tdi= ', final_tdi,
                  ' hausdorff= ', final_hausdorff,
                  ' ramp= ', final_ramp)
            if wandb_logger:
                logger.log({
                    'eval_mse': final_mse,
                    'eval_dtw': final_dtw,
                    'eval_tdi': final_tdi,
                    'eval_hausdorff': final_hausdorff,
                    'eval_ramp': final_ramp
                })
            net.train()

        # Vérification pour early stopping
        if early_stopping:
            # Calcul de la loss sur le jeu de validation selon le loss_type
            net.eval()
            values_loss = []
            with torch.no_grad():
                for val_data in validloader:
                    val_inputs = val_data[0]
                    val_target = val_data[1]
                    val_inputs = val_inputs.to(torch.float32).to(device)
                    val_target = val_target.to(torch.float32).to(device)

                    val_outputs = net(val_inputs)
                    if loss_type == 'mse':
                        val_loss = criterion(val_target, val_outputs)
                    
                    elif loss_type == 'rrmse':
                        mse_loss_val = torch.nn.functional.mse_loss(val_outputs, val_target, reduction='mean')
                        target_mean_squared_val = torch.mean(val_target**2)
                        val_loss = torch.sqrt(mse_loss_val / target_mean_squared_val)

                    elif loss_type == 'dilate':
                        val_loss, _, _ = dilate_loss(val_target, val_outputs, alpha, gamma, device)

                    values_loss.append(val_loss.item())

            current_loss = sum(values_loss)/len(values_loss)
            net.train()
            print("current loss:", current_loss, "best loss :",best_eval_loss)
            if current_loss < best_eval_loss:
                best_eval_loss = current_loss
                epochs_no_improve = 0
                current_best_model = copy.deepcopy(net.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Arrêt anticipé à l'epoch {epoch} car aucune amélioration n'a été constatée durant {patience} évaluations.")
                net.load_state_dict(current_best_model)  # Charger les meilleurs poids dans le modèle
                stop_training = True

        if stop_training:
            break

    if wandb_logger:
        wandb.finish()

# def train_model(net, loss_type, learning_rate, trainloader, validloader, device, data_type, epochs=1000, gamma = 0.001,
#                 print_every=50, verbose=1, alpha=0.5, wandb_logger=False):
#     net.train()
#     optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)
#     criterion = torch.nn.MSELoss()

#     if wandb_logger:
#         if loss_type=="dilate" and alpha>=1-1e-5:
#             name_loss = "sDTW"
#         else:
#             name_loss = loss_type
#         logger = wandb.init(
#             project="DILATE",
#             name=f"model_{data_type}_{gamma}_{alpha}_{name_loss}",  
#         )
    
#     for epoch in range(epochs): 
#         for i, data in enumerate(trainloader, 0):
#             inputs = data[0]
#             target = data[1]
#             inputs = inputs.to(torch.float32).to(device)
#             target = target.to(torch.float32).to(device)
#             batch_size, N_output = target.shape[0:2]                     

#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss_mse,loss_shape,loss_temporal = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            
#             if (loss_type=='mse'):
#                 loss_mse = criterion(target,outputs)
#                 loss = loss_mse

#             elif (loss_type == 'rrmse'):
#                 mse_loss = torch.nn.functional.mse_loss(outputs, target, reduction='mean')
#                 target_mean_squared = torch.mean(target**2)
#                 loss_rrmse = torch.sqrt(mse_loss / target_mean_squared)
#                 loss = loss_rrmse                    
 
#             if (loss_type=='dilate'):    
#                 loss, loss_shape, loss_temporal = dilate_loss(target,outputs,alpha, gamma, device)             
                  
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         if wandb_logger:
#             logger.log({
#                 'train_loss': loss.item(),
#                 'loss_shape': loss_shape.item(),
#                 'loss_temporal': loss_temporal.item(),
#             })          
        
#         if(verbose):
#             if (epoch % print_every == 0):
#                 print('epoch ', epoch, ' loss ',loss.item(),' loss shape ',loss_shape.item(),' loss temporal ',loss_temporal.item())
#                 final_mse, final_dtw, final_tdi, final_hausdorff, final_ramp = eval_model(net, validloader, 
#                                                                                           device=device, data_type=data_type)
#                 print('Eval mse= ', final_mse ,
#                       ' dtw= ', final_dtw ,
#                       ' tdi= ', final_tdi,
#                       ' hausdorff= ', final_hausdorff ,
#                       ' ramp= ', final_ramp)
#                 if wandb_logger:
#                     logger.log({
#                         'eval_mse': final_mse,
#                         'eval_dtw': final_dtw,
#                         'eval_tdi': final_tdi,
#                         'eval_hausdorff': final_hausdorff,
#                         'eval_ramp': final_ramp
#                     }) 
#                 net.train()
#     if wandb_logger:
#         wandb.finish()
  

def eval_model(net, loader, device, data_type):  
    net.eval() 
    criterion = torch.nn.MSELoss()
    losses_mse = []
    losses_dtw = []
    losses_tdi = []   
    losses_hausdorff = []
    losses_ramp = []

    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            loss_mse, loss_dtw, loss_tdi = torch.tensor(0),torch.tensor(0),torch.tensor(0)
            # get the inputs
            inputs = data[0]
            target = data[1]

            inputs = inputs.to(torch.float32).to(device)
            target = target.to(torch.float32).to(device)
            batch_size, N_output = target.shape[0:2]
            outputs = net(inputs)
            
            # MSE    
            loss_mse = criterion(target,outputs)    
            loss_dtw, loss_tdi = 0,0
            # DTW and TDI
            for k in range(batch_size):         
                target_k_cpu = target[k,:,0:1].view(-1).detach().cpu().numpy()
                output_k_cpu = outputs[k,:,0:1].view(-1).detach().cpu().numpy()

                path, sim = dtw_path(target_k_cpu, output_k_cpu)   
                loss_dtw += sim
                        
                Dist = 0
                for i,j in path:
                        Dist += (i-j)*(i-j)
                loss_tdi += Dist / (N_output*N_output)           
                            
            loss_dtw = loss_dtw /batch_size
            loss_tdi = loss_tdi / batch_size

            if data_type=="synthetic":
                bkps = data[2]
                loss_hausdorff = synthetic_hausdorff_distances(inputs, target, bkps)
                losses_hausdorff.append( loss_hausdorff.item() )
            elif data_type=="traffic":
                loss_hausdorff = traffic_hausdorff_distances(inputs, target, outputs)
                losses_hausdorff.append( loss_hausdorff.item() )
            else:
                losses_hausdorff.append(0)

            loss_ramp = ramp_score_batch(true_batch=target, predicted_batch=outputs, epsilon=torch.std(target).item())

            losses_mse.append( loss_mse.item() )
            losses_dtw.append( loss_dtw )
            losses_tdi.append( loss_tdi )
            losses_ramp.append( loss_ramp.item() )
    
    final_mse = np.array(losses_mse).mean()
    final_dtw = np.array(losses_dtw).mean()
    final_tdi = np.array(losses_tdi).mean()
    final_ramp = np.array(losses_ramp).mean()
    final_hausdorff = np.array(losses_hausdorff).mean()

    return final_mse, final_dtw, final_tdi, final_hausdorff, final_ramp


def compare_models(training, net_gru_dilate, net_gru_mse, net_gru_rrmse, net_gru_dtw, trainloader, 
                   validloader, testloader, device, n_epochs, 
                   gamma, alpha, wandb_logger, data, eval_every):
    if training:
        print("-"*130)
        print("TRAINING")
        print("-"*130)
        print("DILATE")
        train_model(
            net_gru_dilate,
            loss_type='dilate',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=n_epochs, 
            gamma=gamma, 
            alpha=alpha,
            verbose=1,
            wandb_logger=wandb_logger,
            data_type=data,
            print_every=eval_every,
            )
        print("-"*130)
        print("MSE")
        train_model(
            net=net_gru_mse,
            loss_type='mse',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=n_epochs, 
            gamma=gamma, 
            alpha=alpha,
            verbose=1,
            wandb_logger=wandb_logger,
            data_type=data,
            print_every=eval_every,
            )
        print("-"*130)
        print("sDTW")
        train_model(
            net=net_gru_dtw,
            loss_type='dilate',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=n_epochs, 
            gamma=gamma, 
            alpha=1,
            verbose=1,
            wandb_logger=wandb_logger,
            data_type=data,
            print_every=eval_every,
            )

        print("-"*130)
        print("RRMSE")
        train_model(
            net=net_gru_rrmse,
            loss_type='rrmse',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=n_epochs, 
            gamma=gamma, 
            alpha=1,
            verbose=1,
            wandb_logger=wandb_logger,
            data_type=data,
            print_every=eval_every,
            )
    
        torch.save(net_gru_dilate.state_dict(), 'weights_models/net_gru_dilate.pth')
        torch.save(net_gru_mse.state_dict(), 'weights_models/net_gru_mse.pth')
        torch.save(net_gru_dtw.state_dict(), 'weights_models/net_gru_dtw.pth')
        torch.save(net_gru_rrmse.state_dict(), 'weights_models/net_gru_rrmse.pth')

    else:
        print("-"*130)
        print("LOADING MODELS")
        net_gru_dilate.load_state_dict(torch.load('weights_models/net_gru_dilate.pth'))
        net_gru_mse.load_state_dict(torch.load('weights_models/net_gru_mse.pth'))
        net_gru_dtw.load_state_dict(torch.load('weights_models/net_gru_dtw.pth'))
        net_gru_rrmse.load_state_dict(torch.load('weights_models/net_gru_rrmse.pth'))

        net_gru_dilate.eval()
        net_gru_mse.eval()
        net_gru_dtw.eval()
        net_gru_rrmse.eval()

    dilate_mse, dilate_dtw, dilate_tdi, dilate_hausdorff, dilate_ramp= eval_model(net_gru_dilate, testloader, device, data_type=data)
    mse_mse, mse_dtw, mse_tdi, mse_hausdorff, mse_ramp = eval_model(net_gru_mse, testloader, device, data_type=data)
    dtw_mse, dtw_dtw, dtw_tdi, dtw_hausdorff, dtw_ramp = eval_model(net_gru_dtw, testloader, device, data_type=data)
    rrmse_mse, rrmse_dtw, rrmse_tdi, rrmse_hausdorff, rrmse_ramp = eval_model(net_gru_rrmse, testloader, device, data_type=data)
    
    print("-"*130)
    print("EVALUATION")
    print("-"*130)
    print("Eval dilate")
    print('mse= ', dilate_mse ,
        ' dtw= ', dilate_dtw ,
        ' tdi= ', dilate_tdi,
        ' hausdorff= ', dilate_hausdorff ,
        ' ramp= ', dilate_ramp) 
    print("-"*130)
    print("Eval mse")
    print('mse= ', mse_mse ,
        ' dtw= ', mse_dtw ,
        ' tdi= ', mse_tdi,
        ' hausdorff= ', mse_hausdorff ,
        ' ramp= ', mse_ramp) 
    print("-"*130)
    print("Eval softDTW")
    print('mse= ', dtw_mse ,
        ' dtw= ', dtw_dtw ,
        ' tdi= ', dtw_tdi,
        ' hausdorff= ', dtw_hausdorff ,
        ' ramp= ', dtw_ramp) 
    print("-"*130)
    print("Eval RRMSE")
    print('mse= ', rrmse_mse ,
        ' dtw= ', rrmse_dtw ,
        ' tdi= ', rrmse_tdi,
        ' hausdorff= ', rrmse_hausdorff ,
        ' ramp= ', rrmse_ramp) 
    print("-"*130)


def compare_gammas(gammas, output_length, device, batch_size, trainloader, 
                   validloader, testloader, n_epochs, wandb_logger, data, eval_every):
    print("-" * 130)
    print("TRAINING FOR DIFFERENT GAMMAS")
    
    metrics = {
        "gamma": [],
        "mse": [],
        "dtw": [],
        "tdi": [],
        "hausdorff": [],
        "ramp": []
    }

    for gamma in gammas:
        encoder_dtw = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
        decoder_dtw = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, fc_units=16, output_size=1).to(device)
        net_gru_dtw = Net_GRU(encoder_dtw, decoder_dtw, output_length, device).to(device)

        print("-" * 130)
        print(f"sDTW with gamma={gamma}")
        train_model(
            net=net_gru_dtw,
            loss_type='dilate',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=n_epochs, 
            gamma=gamma, 
            alpha=1,
            verbose=1,
            wandb_logger=wandb_logger,
            data_type=data,
            print_every=eval_every,
        )
        
        dtw_mse, dtw_dtw, dtw_tdi, dtw_hausdorff, dtw_ramp = eval_model(net_gru_dtw, testloader, device, data_type=data)

        metrics["gamma"].append(gamma)
        metrics["mse"].append(dtw_mse)
        metrics["dtw"].append(dtw_dtw)
        metrics["tdi"].append(dtw_tdi)
        metrics["hausdorff"].append(dtw_hausdorff)
        metrics["ramp"].append(dtw_ramp)

        print("-" * 130)
        print(f"Eval softDTW with gamma={gamma}")
        print('mse= ', dtw_mse,
              ' dtw= ', dtw_dtw,
              ' tdi= ', dtw_tdi,
              ' hausdorff= ', dtw_hausdorff,
              ' ramp= ', dtw_ramp) 

        del encoder_dtw, decoder_dtw, net_gru_dtw
        torch.cuda.empty_cache()

    print("-" * 130)
    return metrics


def compare_alphas(alphas, gamma, output_length, device, batch_size, trainloader, 
                   validloader, testloader, n_epochs, wandb_logger, data, eval_every):
    print("-" * 130)
    print("TRAINING FOR DIFFERENT ALPHAS")
    
    metrics = {
        "alpha": [],
        "mse": [],
        "dtw": [],
        "tdi": [],
        "hausdorff": [],
        "ramp": []
    }

    for alpha in alphas:
        encoder_dilate = EncoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, batch_size=batch_size).to(device)
        decoder_dilate = DecoderRNN(input_size=1, hidden_size=128, num_grulstm_layers=1, fc_units=16, output_size=1).to(device)
        net_gru_dilate = Net_GRU(encoder_dilate, decoder_dilate, output_length, device).to(device)

        print("-" * 130)
        print(f"DILATE with alpha={alpha}")
        train_model(
            net=net_gru_dilate,
            loss_type='dilate',
            learning_rate=0.001,
            trainloader=trainloader,
            validloader=validloader,
            device=device,
            epochs=n_epochs, 
            gamma=gamma, 
            alpha=alpha,
            verbose=1,
            wandb_logger=wandb_logger,
            data_type=data,
            print_every=eval_every,
        )
        
        dilate_mse, dilate_dtw, dilate_tdi, dilate_hausdorff, dilate_ramp = eval_model(net_gru_dilate, 
                                                                                       testloader, 
                                                                                       device, 
                                                                                       data_type=data)

        metrics["alpha"].append(alpha)
        metrics["mse"].append(dilate_mse)
        metrics["dtw"].append(dilate_dtw)
        metrics["tdi"].append(dilate_tdi)
        metrics["hausdorff"].append(dilate_hausdorff)
        metrics["ramp"].append(dilate_ramp)

        print("-" * 130)
        print(f"Eval DILATE with alpha={alpha}")
        print('mse= ', dilate_mse,
              ' dtw= ', dilate_dtw,
              ' tdi= ', dilate_tdi,
              ' hausdorff= ', dilate_hausdorff,
              ' ramp= ', dilate_ramp) 
        
        del encoder_dilate, decoder_dilate, net_gru_dilate
        torch.cuda.empty_cache()

    print("-" * 130)
    return metrics
