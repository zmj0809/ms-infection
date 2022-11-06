### Reqiurements
torch                   1.12.1+cu116
pandas                  1.3.5
numpy                   1.21.6
scikit-learn            0.24.2
matplotlib              3.5.3
tqdm                    4.64.0

### Training 
Please run the file ```python clf_dl.py``` in the command line.

#### Parameters
--task: perform classification by metabolic data or proteomic data; default 
    parser.add_argument("--file_marker", type = str, default = "forInfection")
    parser.add_argument("--epochs", type = int, default = 200)

    parser.add_argument("--hid_dim", type = int, default = -1)
    parser.add_argument("--nlayer", type = int, default = -1)
    parser.add_argument("--drop", type = float, default = -1)

    parser.add_argument("--device_id", type = int, default = 0)
clf_dl.py will perform the parameter optimization automatically. The model with best performance in validation set will be chosen as the optimized model for classification

**Attention: Due to different softwares (e.g., operation system) and different hardwares (e.g., GPU devices), the optimized parameter will be different. But the classification performance is close.**

### Testing
Please run the file ```python ms_dl.py``` in the command line.