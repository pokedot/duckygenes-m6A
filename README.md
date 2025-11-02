# duckygenes-m6A
This project focuses on site-level m6A modification detection from nanopore/RNA-seq read-level features using a Transformer-based MIL (Multiple Instance Learning) model.

## Repo layout
```
.
└── duckygenes-m6A/
    ├── data/                                <- Sample test dataset to run the python script
    │   └── dataset2.json.gz  
    ├── models/                              <- Saved models and artifacts from training
    │   ├── amp_grad_scaler.pt
    │   ├── best_model_state.pt
    │   ├── final_checkpoint.pt
    │   ├── final_model_state_dict.pt
    │   └── scaler.joblib
    ├── notebooks/                           <- Notebooks created during the project experimentation
    │   ├── cnn_mil_transformers_inference.ipynb
    │   └── cnn_mil_transformers_training.ipynb
    ├── predictions/                         <- Output directory from model inference
    ├── scripts/                             <- Finalised python scripts to run the model
    │   └── cnn_mil_transformers_inference.py  (Run this!)
    ├── README.md
    └── requirements.txt                     <- List of packages required for this project
```

## Quick start (inference)

**Note: We recommend using a machine with sufficient storage for this task (e.g. t3.large).**<br>
If you do not have Python/ Pip installed:
1. Run the following command in the Ubuntu terminal: `sudo apt install python3 python3-pip`

Once Python is installed:
1. Run the following command in the Ubuntu terminal: `git clone https://github.com/pokedot/duckygenes-m6A.git`
2. Then, navigate into the project folder: `cd duckygenes-m6A`
3. Next, install all the required packages: `pip3 install -r requirements.txt`
4. Navigate into the scripts folder: `cd scripts`
5. Finally, run the .py script. Note that the script uses dataset2.json.gz by default: `python3 cnn_mil_transformers_inference.py`<br>
*You may choose to run the script on a different dataset. Upload your dataset into the data folder, then edit the .py script's parameters to your dataset file name.*

## File formats and artifacts

- Checkpoint: a PyTorch state dict (OrderedDict) with parameter keys. Training may save full checkpoint dicts (inspect top-level keys like `model_state_dict`).
- Feature scaler: recommended `joblib.dump({'scaler': scaler, 'num_cols': num_cols}, 'scaler.joblib')` to store both the scaler and the expected numeric column order.
- Input: gzipped JSON mapping transcript IDs → positions → sequences → list of read-level numeric vectors (the notebooks contain the loader).
- Output: CSV with columns `transcript_id, transcript_position, score` (per-site probability).

## Dependencies
This project's required packages are specified in the requirements.txt file found at the root of this repository. Before running the inference script, run `pip install -r requirements.txt` to install them.

## Acknowledgements
All Nanopore long read RNA-Seq data are obtained from the [SG-NEx project](https://github.com/GoekeLab/sg-nex-data).

## License & contact
This repository includes a LICENSE file — please review for reuse terms.
