# Name Entity Recognition problem

## Task decribtion
In this task, we need to train a named entity recognition (NER) model for the identification of
mountain names inside the texts. For this purpose you need:
- Find / create a dataset with labeled mountains.
- Select the relevant architecture of the model for NER solving.
- Train / finetune the model.
- Prepare demo code / notebook of the inference results.
The output for this task should contain:
- Jupyter notebook that explains the process of the dataset creation.
- Dataset including all artifacts it consists of.
- Link to model weights.
- Python script (.py) for model training.
- Python script (.py) for model inference.
- Jupyter notebook with demo.

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alexmarmaliuk/quantum_task.git

2. **Navigate to the project folder:**
    ```bash
    cd repository-name

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt

## Task summary

```
NER/\\
├── data/\\
│   └── gpt_dataset.csv\\
├── utils/\\
│   └── plotting.py\\
├── demo.ipynb\\
├── model.py\\
├── README.md\\
├── report.pdf\\
├── requirements.txt\\
└── rnn_model.pth
```

- 'data' folder contains 'gpt_dataset.csv' file with  a dataset, generated via ChatGPT. It is a tabular annotated sequence of words, arranged in sentences, each ending with a fullstop.
- 'utils' folder contains files with custom supplementary functions, aimed at private use.
- demo.ipynb is a main demonstration
- model.py is a separate file, which contains definition of the model class
- rnn_model.pth is a trained model, saved for potential future use