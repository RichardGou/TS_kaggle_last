# DILATE

## Paramètres:
- data : `--data synthetic` or `--data traffic`, if `--data traffic`, must add path to data `--path_traffic "/kaggle/input/traffic/traffic.txt"` for example.
- visualization: `--viz` or `no-viz`
- number of epochs: `--n_epochs 2`

## Experiment Options:

### 1. Finding the Optimal Gamma

To identify the best gamma for the softDTW loss, run the following command:
`python main.py --data synthetic --n_epochs 5 --gamma choice --viz`
This will compare the performance of the model using the softDTW loss with various gamma values.

### 2. Finding the Optimal Alpha

Once the optimal gamma is determined for the chosen dataset, run an experiment to find the best alpha:
`python main.py --data synthetic --n_epochs 5 --gamma 0.01 --alpha choice --viz`
This will compare the performance of the model using the DILATE loss with different alpha values.

### 3. Comparing Models with Fixed Gamma and Alpha

After determining the best alpha and gamma, compare the DILATE model with models trained using other metrics. Run the following command:
`python main.py --data synthetic --n_epochs 5 --gamma 0.01 --alpha 0.2 --viz`

If the models are already trained on this dataset, you can reuse them to save time by disabling training with the following parameter:
training: `--train` or `no-train`


"# TS_kaggle_last" 
