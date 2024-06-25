# Local-LLM-Llama3

This repository contains scripts, notebooks, and utilities for training and running inference on language models, with a focus on Amharic language processing.

## Project Structure

- **notebooks/**: Jupyter notebooks for experimentation and analysis.
  - `1.0-amharic-tokenization.ipynb`: Notebook for tokenization of Amharic text.
  - `2.0-continued-pretraining-unsloth.ipynb`: Notebook for continued pretraining using the Unsloth framework.

- **scripts/**: Shell scripts for running training and inference.
  - `run_hf_trainer.sh`: Script to run the Hugging Face trainer.
  - `run_unsloth_trainer.sh`: Script to run the Unsloth trainer.
  - `run_inference.sh`: Script to perform inference.

- **src/**: Source code for data processing, training, and inference.
  - `data_processing.py`: Utilities for preparing datasets.
  - `tokenization.py`: Tokenization logic for Amharic text.
  - `hf-trainer.py`: Hugging Face trainer implementation.
  - `unsloth-trainer.py`: Unsloth trainer implementation.
  - `inference.py`: Inference logic for trained models.
  - `utils.py`: Miscellaneous utility functions.

- **requirements.txt**: Python dependencies for the project.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook
- Required Python packages (install via `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/Local-LLM-Llama3.git
   cd Local-LLM-Llama3
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Training

To train a model using the Hugging Face trainer:
```bash
bash scripts/run_hf_trainer.sh
```

To train a model using the Unsloth trainer:
```bash
bash scripts/run_unsloth_trainer.sh
```

#### Inference

To run inference:
```bash
bash scripts/run_inference.sh
```
