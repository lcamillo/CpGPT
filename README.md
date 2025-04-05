<div align="center">

# CpGPT: A Foundation Model for DNA Methylation

<img src="cpgpt_logo.svg" width="200px" alt="CpGPT Logo">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI](https://img.shields.io/pypi/v/CpGPT.svg)](https://pypi.org/project/CpGPT/)
[![PyTorch 2.5+](https://img.shields.io/badge/torch-2.5+-ee4c2c.svg)](https://pytorch.org/get-started/locally/)
[![Lightning 2.5+](https://img.shields.io/badge/lightning-2.5+-792ee5.svg)](https://lightning.ai/)
[![License](https://img.shields.io/badge/License-Non_Commercial-red.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/bioRxiv-2024.10.24.619766-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1)

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&pause=1000&color=2E98FF&center=true&vCenter=true&width=600&lines=A+foundation+model+for+DNA+methylation;Generate%2C+impute%2C+and+embed+methylation+profiles;Fine-tune+for+epigenetics+and+aging+research;CpGCpGCpGCpGCpGCpGCpGCpGCpGCpGCpGCpG)](https://github.com/lcamillo/CpGPT)

</div>

**EXTREMELY IMPORTANT NOTE ABOUT DATA**: The data for CpGPT is hosted on an AWS S3 bucket with a requester-pays configuration. This means that <u>you will be charged</u> by AWS for the data you download. **The total cost for downloading all datasets will *exceed* <u>$150.00 USD</u> at the time of writing.** We will do our best to estimate the cost of downloading for you, but we cannot guarantee the accuracy of our estimate. To estimate the cost of downloading part of the bucket, you can use the following command:

```bash
S3_BUCKET='s3://cpgpt-lucascamillo-public'

# Replace <PREFIX> with the s3 path you want to estimate the cost for example:
BUCKET_PREFIX="${S3_BUCKET}/<PREFIX>/"
# For example: BUCKET_PREFIX="${S3_BUCKET}/data/cpgcorpus/raw/"

# To estimate the size of data and cost before downloading, use:
TOTAL_SIZE=$(aws s3 ls --recursive --human-readable --summarize --request-payer requester $BUCKET_PREFIX | grep "Total Size")

# Likely cost is $0.09 per GB downloaded.
TOTAL_SIZE_GB=$(grep -oP 'Total Size: \K[0-9.]+' <<< "$TOTAL_SIZE")
EXPECTED_COST=$(echo "$TOTAL_SIZE_GB" | python -c "print(float(input()) * 0.09)")

# Print the estimated cost
echo "Estimated cost to download all data: \$$EXPECTED_COST USD for $TOTAL_SIZE_GB GB"
```


## 📋 Table of Contents

- [📖 Overview](#-overview)
- [🚀 Quick Setup](#-quick-setup)
- [🗄️ CpGCorpus](#%EF%B8%8F-cpgcorpus)
- [🐘 Model Zoo](#-model-zoo)
- [🧪 Tutorials](#-tutorials)
- [🔧 Finetuning](#-finetuning)
- [❓ FAQ](#-faq)
- [📚 Citation](#-citation)
- [☎️ Contact](#-contact)
- [📜 License](#-license)

## 📖 Overview

CpGPT is a foundation model for DNA methylation, trained on genome-wide DNA methylation data. It can generate, impute, and embed methylation profiles, and can be finetuned for various downstream tasks.

## 🚀 Quick Setup

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- AWS CLI (for downloading dependencies)

### Installation Instructions

Typical installation (`pip`):

```bash
pip install cpgpt
```

For local installation:

```bash
# Clone the repository
git clone https://github.com/lcamillo/CpGPT.git
cd CpGPT

# Install poetry if you don't already have it
curl -sSL https://install.python-poetry.org | python3 -

# Install conda if you don't already have it
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh

# Create a conda environment
conda create -y -n cpgpt python=3.12 pygraphviz
conda activate cpgpt

# Install dependencies in conda environment
poetry install
```

Alternatively, the package is available through:

```bash
# Install with pip
pip install CpGPT
```

### Setting up AWS CLI for Dependencies

Our pre-trained models and data are stored in AWS S3. If you do not already have an AWS account setup, follow these steps:

<details closed>
<summary><b>1. Create an AWS Account</b></summary>

1. Go to [AWS Console](https://aws.amazon.com/) and click "Create an AWS Account" in the top right
2. Follow the signup process:
   - Provide email and account name
   - Enter your personal/business information
   - Add payment information (a credit card is required, but the downloads follow free tier limits)
   - Complete identity verification (you'll receive a phone call or text)
   - Select a support plan (Free tier is sufficient)

</details>

<details closed>
<summary><b>2. Install the AWS CLI</b></summary>

**For Linux/macOS:**

```bash
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
```

**For Windows:**

- Download the [AWS CLI MSI installer](https://awscli.amazonaws.com/AWSCLIV2.msi)
- Run the downloaded MSI installer and follow the on-screen instructions

**Verify installation:**

```bash
aws --version
```

</details>

<details closed>
<summary><b>3. Create Access Keys</b></summary>

1. Log in to the [AWS Console](https://console.aws.amazon.com/)
2. Click on your account name in the top right, then "Security credentials"
3. Scroll down to "Access keys" and click "Create access key"
4. Select "Command Line Interface (CLI)" as the use case
5. Check the "I understand..." acknowledgment and click "Next"
6. **IMPORTANT**: Download the CSV file or copy both the "Access key ID" and "Secret access key" to a secure location. You will not be able to view the secret access key again.

</details>

<details closed>
<summary><b>4. Configure AWS CLI</b></summary>

Run the following command and enter your credentials when prompted:

```bash
aws configure
```

You'll need to input:

- **AWS Access Key ID**: The access key ID from step 3
- **AWS Secret Access Key**: The secret access key from step 3
- **Default region name**: Enter `us-east-1` (where our data is hosted)
- **Default output format**: Enter `json`

</details>

<details closed>
<summary><b>5. Test Your Configuration</b></summary>

Verify your setup with this command that lists the contents (without downloading):

```bash
aws s3 ls --request-payer requester s3://cpgpt-lucascamillo-public/data/cpgcorpus/raw/
```

You should see a list of GSE folders if your configuration is correct.

</details>

## 🗄️ CpGCorpus

<details closed>
<summary><b>Download the Full Corpus</b></summary>

To download the entire CpGCorpus from our S3 bucket, run the following command. **SINCE YOU ARE USING A REQUESTER-PAYS-FOR-DATA CONFIGURATION, YOU WILL BE CHARGED BY AWS FOR THE DATA YOU DOWNLOAD, WHICH IS <u>~$100.00 USD</u> AT THE TIME OF WRITING.** 

```bash
aws s3 sync s3://cpgpt-lucascamillo-public/data/cpgcorpus/raw ./data/cpgcorpus/raw --request-payer requester
```

</details>

<details closed>
<summary><b>Directory Layout</b></summary>

The CpGCorpus is organized in a hierarchical structure by GSE (Gene Series) and further by GPL (Platform). Below is an overview of the directory layout and file contents:

```
cpgcorpus/
  └── raw/
      └── {GSE_ID}/
          └── {GPL_ID}/
              ├── betas/
              │   ├── QCDPB.arrow      # Processed beta values via the R sesame QCDPB pipeline
              │   └── gse_betas.arrow  # Raw beta values downloaded from GEO
              └── metadata/
                  └── metadata.arrow   # Metadata and sample annotations
```

- The "betas" folder contains one of the two files:
  - QCDPB.arrow: Processed data from the R sesame QCDPB pipeline.
  - gse_betas.arrow: Beta values as originally downloaded from GEO.
- The "metadata" folder stores the metadata.arrow file that holds supplementary experimental details.

</details>

<details closed>
<summary><b>Supported Methylation Platforms</b></summary>

The corpus includes multiple platforms:

- GPL8490 (27k array)
- GPL13534 (450k)
- GPL18809 (450k)
- GPL21145 (EPIC)
- GPL23976 (EPIC)
- GPL29753 (EPIC)
- GPL33022 (EPICv2)
- GPL34394 (MSA)

</details>

<details closed>
<summary><b>Download a specific sample</b></summary>

To download a specific dataset (for example, GSE163839 using platform GPL13534), run:

```bash
aws s3 cp s3://cpgpt-lucascamillo-public/data/cpgcorpus/raw/GSE163839/GPL13534/betas/QCDPB.arrow ./data/GSE163839.arrow --requester-payer requester
```

</details>

## 🐘 Model Zoo

There are several versions of CpGPT, mainly divided into pretrained and finetuned models. Below, you can find a table with a summary of such versions including the model name for download.

<details open>
<summary><b>Pre-trained Models</b></summary>

| Model      | Size  | Parameters | Description                                                                       | Model Name |
| ---------- | ----- | ---------- | --------------------------------------------------------------------------------- | ---------- |
| CpGPT-2M   | 30MB  | ~2.5M      | Lightweight model for quick experimentation and resource-constrained environments | `small`    |
| CpGPT-100M | 1.1GB | ~101M      | Full-size model for state-of-the-art performance and high accuracy                | `large`    |

</details>

<details>
<summary><b>Fine-tuned Models</b></summary>

> ⚠️ **Note**: Fine-tuned model weights are currently being updated and will be available soon. The table below shows the models that will be provided.

We provide specialized pre-trained models for common tasks:

| Model                       | Parameters | Description                                              | Output                                                                 | Model Name            |
| --------------------------- | ---------- | -------------------------------------------------------- | ---------------------------------------------------------------------- | --------------------- |
| CpGPT-2M-Age                | ~2.9M      | Multi-tissue chronological age predictor                 | Age in years                                                           | `age`                 |
| CpGPT-2M-AverageAdultWeight | ~2.9M      | Multi-tissue, pan-mammalian weight predictor             | Log1p of average adult weight in kilograms                             | `average_adultweight` |
| CpGPT-100M-BoA              | ~101M      | EPICv2 blood imputation                                  | No phenotype is predicted                                              | `boa`                 |
| CpGPT-2M-Cancer             | ~2.9M      | Multi-tissue cancer predictor                            | Logits of cancer status (use sigmoid to get probabilities)             | `cancer`              |
| CpGPT-2M-ClockProxies       | ~3.1M      | Blood proxies of five epigenetic clocks                  | altumage, dunedinpace (x100), grimage2, hrsinchphenoage, pchorvath2013 | `clock_proxies`       |
| CpGPT-2M-EpicMammal         | ~2.5M      | Blood EPIC-Mammalian array converter                     | No phenotype is predicted                                              | `epicvmammal`         |
| CpGPT-100M-Hannum           | ~101M      | 450k blood imputation                                    | No phenotype is predicted                                              | `hannum`              |
| CpGPT-100M-HumanRRBSAtlas   | ~101M      | Multi-tissue RRBS imputation                             | No phenotype is predicted                                              | `human_rrbs_atlas`    |
| CpGPT-100M-Mammalian        | ~101M      | Multi-tissue, pan-mammalian mammalian array imputation   | No phenotype is predicted                                              | `mammalian`           |
| CpGPT-2M-MaxLifespan        | ~2.9M      | Multi-tissue, pan-mammalian max lifespan predictor       | Log1p of max lifespan in years                                         | `maximum_lifespan`    |
| CpGPT-2M-Mortality          | ~2.9M      | Blood mortality predictor. Please use strict_load=False. | Risk score                                                             | `mortality`           |
| CpGPT-2M-RelativeAge        | ~2.9M      | Multi-tissue, pan-mammalian relative age predictor       | Relative age (0 to 1)                                                  | `relative_age`        |
| CpGPT-100M-sciMETv3         | ~101M      | Brain, single-cell imputation                            | No phenotype is predicted                                              | `scimetv3`            |

</details>

## 🧪 Tutorials

More tutorials will be added soon!

<div class="tutorial-cards" style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px;">
  <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; width: 250px;">
    <h3>🔬 Quick setup</h3>
    <p>Basic introduction to CpGPT and its capabilities</p>
    <a href="tutorials/quick_setup.ipynb">View Tutorial</a>
  </div>
  <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; width: 250px;">
    <h3>🗺️ Reference map</h3>
    <p>Zero-shot label transfer to a reference dataset</p>
    <a href="tutorials/reference_map.ipynb">View Tutorial</a>
  </div>
</div>

## 🔧 Finetuning

> ⚠️ **Warning**: Fine-tuning CpGPT models requires a GPU. The training process is computationally intensive and will be extremely slow or may fail entirely without GPU acceleration. We recommend at least 8GB of VRAM for the small model and 24GB+ for the large model.

<details closed>
<summary><b>Getting Started</b></summary>

1. **Download dependencies** if you have not already done so by following the steps in the <a href="tutorials/quick_setup.ipynb">quick setup tutorial notebook</a>.

2. **Prepare your data** by following the steps in the <a href="tutorials/quick_setup.ipynb">quick setup tutorial notebook</a>.

</details>

<details closed>
<summary><b>Configuration</b></summary>

1. **Create a configuration file** by modifying template in `configs/experiment/`.

2. **Run fine-tuning** with the CLI:

```bash
cpgpt-train experiment=template
```

3. **Get the best checkpoint** in the logs folders:

- **Checkpoint weights**: `logs/experiment/{experiment_name}/checkpoints/{experiment_name}.ckpt`
- **Model config**: `logs/experiment/{experiment_name}/.hydra/config.yaml`

</details>

### Configuration Guide

<details>
<summary><b>🔍 Model Configuration</b></summary>

CpGPT provides several parameters to customize your model architecture and training process:

| Parameter         | Description             | Examples                                           |
| ----------------- | ----------------------- | -------------------------------------------------- |
| `model/net`       | Model architecture size | `small.yaml`, `large.yaml`                         |
| `model/optimizer` | Optimization algorithm  | `adamw.yaml`, `adamwscheduleree.yaml`, `lion.yaml` |
| `model/scheduler` | Learning rate scheduler | `cosine_warmup.yaml`, `constant.yaml`              |

</details>

<details>
<summary><b>📊 Task-Specific Settings</b></summary>

Modify these parameters in your experiment YAML file to customize the model for different tasks:

```yaml
model:
  training:
    # Type of loss function for condition decoder
    condition_decoder_loss: mae  # Options: mae, mse, ce

    # Weighting for the condition loss vs reconstruction
    loss_weights:
      condition_loss: 0.1

  optimizer:
    # Learning rate
    lr: 0.0001

  net:
    # Enable the condition decoder for prediction tasks
    use_condition_decoder: true

    # Number of target variables to predict
    condition_size: 1  # 1 for regression, can be >1 for multi-target
```

</details>

<details>
<summary><b>⚙️ Training Parameters</b></summary>

Control the training process with these settings:

```yaml
trainer:
  # Minimum training steps (for warmup)
  min_steps: 2000

  # Maximum training steps before stopping
  max_steps: 100000

data:
  # Batch size for training
  batch_size: 16  # Reduce for large models or limited GPU memory

  # Data directories
  train_dir: ${paths.data_dir}/mydata/processed/train
  val_dir: ${paths.data_dir}/mydata/processed/val
  test_dir: ${paths.data_dir}/mydata/processed/test
```

</details>

<details>
<summary><b>💾 Checkpointing</b></summary>

Configure model saving behavior:

```yaml
callbacks:
  model_checkpoint:
    # Metric to monitor for saving best model
    monitor: "val/condition_loss"  # Options: val/loss, val/condition_loss, etc.

    # Filename pattern for saved checkpoints
    filename: "${tags[0]}"  # Uses the first tag as filename

    # Save mode
    mode: "min"  # min for losses, max for metrics like accuracy
```

</details>

<details>
<summary><b>📝 Logging</b></summary>

Configure experiment logging with these options:

```yaml
logger:
  # WandB logging
  wandb:
    project: "cpgpt"
    name: "${tags[0]}"
    tags: ${tags}
    group: "${task_name}"

  # TensorBoard logging
  tensorboard:
    name: "tensorboard"
    save_dir: "logs/tensorboard/"

  # CSV logging
  csv:
    name: "csv"
    save_dir: "logs/csv/"
```

Available loggers include:

- `wandb.yaml`: Weights & Biases for experiment tracking with visualization
- `tensorboard.yaml`: TensorBoard for local visualization
- `csv.yaml`: Simple CSV logging for offline analysis
- `mlflow.yaml`: MLflow for organization-level experiment tracking

</details>

## ❓ FAQ

<details>
<summary><b>What methylation array platforms are supported?</b></summary>
CpGPT was pretrained with bulk data from all of the available Illumina arrays, besides the Horvath Mammalian array, at the time of writing. Nevertheless, CpGPT should be able to generalize to new arrays and unseen genomic loci. For RRBS and other types of sequencing-based methylation measurements, finetuning with at least a subset of the data is highly recommended.
</details>

<details>
<summary><b>How much data do I need to fine-tune CpGPT?</b></summary>
CpGPT can be fine-tuned with as few as 50-100 samples for simple tasks. For complex tasks or higher accuracy, we recommend 500+ samples.
</details>

<details>
<summary><b>Should I filter the CpG sites prior to finetuning?</b></summary>
That depends on the task and the reason for finetuning. For instance, to finetune for a model that does not predict specific phenotypes and is just used to learn whole-genome methylation profiles, then it is best not to filter any features. However, if there is a specific phenotype to be predicted, then using a ridge regression and picking the top N features can speed up the training time required (see below).
</details>

<details>
<summary><b>How many steps should I finetune it for?</b></summary>
That ultimately depends on how many samples and how many features are shown to the model. As a rough guide, showing CpGPT each sample-feature combination 50 times works well. For instance, if there are 100 samples with 10,000 CpG sites each, then with a batch size of 10, 100,000 steps would be ideal.
</details>

<details>
<summary><b>How can I get the very best possible performance?</b></summary>
One trick that can increase training time substantially but can lead to some minor performance improvements is to change the following parameter in the `template.yaml` file:

```yaml
model:
  training:
    generative_splits: 5
```

The default for that parameter is 2, which effectively means that generative training is not used.

</details>

<details>
<summary><b>Can I use CpGPT for commercial purposes?</b></summary>
The current release is for non-commercial research purposes only. Please contact us for licensing information for commercial use.
</details>

## 📚 Citation

If you use CpGPT in your research, please cite our paper:

```bibtex
@article{camillo2024cpgpt,
  title={CpGPT: A Foundation Model for DNA Methylation},
  author={de Lima Camillo, Lucas Paulo et al.},
  journal={bioRxiv},
  year={2024},
  doi={10.1101/2024.10.24.619766},
  url={https://www.biorxiv.org/content/10.1101/2024.10.24.619766v1}
}
```

## ☎️ Contact

For contact, please email lucas_camillo@alumni.brown.edu.

## 📜 License

This project is licensed for non-commercial research purposes only. See [LICENSE](LICENSE) for details.

______________________________________________________________________

<div align="center">
  <p>© 2024 Lucas Paulo de Lima Camillo</p>
  <a href="https://twitter.com/lucascamillomd"><img src="https://img.shields.io/twitter/follow/lucascamillomd?style=social" alt="Twitter Follow"></a>
</div>
