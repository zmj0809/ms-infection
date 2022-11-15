## Data
### Path
The metabolic and proteomic MS data can be found in the folder ```data/```.
- **met_forInfection.csv**: the metabolic data for infection classification
- **pro_forInfection.csv**: the proteomic data for infection classification
- **met_forVirus.csv**: the metabolic data for virus infection classification
- **pro_forVirus.csv**: the proteomic data for virus infection classification

## Saved models and predictions
### Path
The metabolic and proteomic MS data can be found in the folder ```models/```.
### Saved models for infection classification 
- **dl_forInfection_both**: trained with bimodal of metabolic and proteomic MS data
- **dl_forInfection_met**: trained with single modal of metabolic MS data
- **dl_forInfection_pro**: trained with single modal of proteomic MS data
### Saved models for virus infection classification 
- **dl_forVirus_both**: trained with bimodal of metabolic and proteomic MS data
- **dl_forVirus_met**: trained with single modal of metabolic MS data
- **dl_forVirus_pro**: trained with single modal of proteomic MS data

## Codes
### Reqiurements
```
torch                   1.12.1+cu116
pandas                  1.3.5
numpy                   1.21.6
scikit-learn            0.24.2
matplotlib              3.5.3
tqdm                    4.64.0
```
### Performance validation
Please run the file ```python test_dl.py``` in the command line to test the reported model in our manuscript. 

* **task**: perform classification by metabolic data or proteomic data; 
        default = ```both```; choose from ```met```, ```pro```, and ```both```
* **file_marker**: perform classification of which infection subtypes; default = ```forInfection```; choose from ```forInfection``` and ```for Virus```
* **hid_dim**: the number of hidden units, which will be optimized by the performance of validation set
* **nlayer**: the number of residual blocks, which will be optimized by the performance of validation set
* **drop**: dropout rate, which will be optimized by the performance of validation set
* **device_id**: run the file on which GPU device 
