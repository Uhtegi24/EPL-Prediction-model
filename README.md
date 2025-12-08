# EPL-Prediction-model
EPL Match Outcome Prediction model - Märten Ühtegi, Tristan Imala, Karl Johannes Teetlaus


## Motivation

The motivation for the project mainly derived from our interest in football and the premise of sport betting. So the end goal was to write a programme that would analyse past matches and make a compotent prediction on the outcome of a match.

## Install Dependencies

After cloning the repository, you must install the required Python packages
before running the notebook or the model.

From the project root, run:

```bash
pip install -r requirements.txt
```

## Code replication
All analyses presented in this project are fully reproducible using the provided source code. The raw match data is loaded and preprocessed using the data processing pipeline (data.py), which generates a standardized match-level dataset. Model training and evaluation are then performed through the unified training function in model.py, allowing the same Random Forest analysis to be replicated by running the corresponding notebook or command-line script. By following the documented workflow and installing the listed dependencies, another user can reproduce the results and extend the analysis with minimal additional configuration.
