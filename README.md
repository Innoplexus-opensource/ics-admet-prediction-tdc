# ADMET Property Classification Using Attention-Based Deep Neural Networks

## Overview
This project implements a deep learning model for predicting ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) properties of chemical compounds. The model utilizes an attention-based neural network architecture and Morgan fingerprints for molecular representation.

## Author Information
- **Author**: Rohit Yadav
- **Organization**: Innoplexus Consulting Services Pvt. Ltd., A Partex Company

## Features
- Attention-based neural network architecture
- Support for both binary classification and regression tasks
- Morgan fingerprint-based molecular representation
- Comprehensive evaluation metrics
- Multiple ADMET property predictions
- 5-fold cross-validation

## Dependencies
- Python 3.7+
- PyTorch
- RDKit
- NumPy
- Pandas
- Scikit-learn
- PyTDC (Therapeutics Data Commons)

## Installation

## Setup
Setup the environment for the project using `environment.yml` by running following command.

```
conda env create --name admet --file=environment.yml
conda activate admet
pip install -r requirements.txt 
```
Alternatively, the environment can be setup using following steps:

```
pip install torch rdkit numpy pandas scikit-learn PyTDC 

```

## Supported ADMET Properties
1. **Absorption**
   - Caco2 permeability
   - Bioavailability
   - Human Intestinal Absorption (HIA)

2. **Distribution**
   - Blood-Brain Barrier penetration
   - Plasma Protein Binding
   - Volume of Distribution

3. **Metabolism**
   - CYP450 inhibition (2C9, 2D6, 3A4)
   - CYP450 substrate specificity

4. **Excretion**
   - Clearance (Hepatocyte and Microsome)
   - Half-life

5. **Toxicity**
   - hERG inhibition
   - AMES mutagenicity
   - Drug-induced liver injury (DILI)
   - LD50
   - Carcinogenicity
   - Skin reaction

## Usage
```python
python admet_prediction.py
```

## Output Files
-  `benchmark_results.log` : Detailed logging of model performance
-  `benchmark_log.txt` : Summary of benchmark results
-  `saved_models/` : Directory containing trained models

## Contributing
Please feel free to submit issues and pull requests.


## Contact
Please contact [Rohit Yadav](rohit.yadav@ics.innoplexus.com) if you have any questions!

Feel free to contribute to this project by opening issues or submitting pull requests. For any questions or inquiries, please contact the repository maintainer.

## Powered By

<img src="https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/_jcr_content/root/responsivegrid/nv_container_392921705/nv_container/nv_image.coreimg.100.630.png/1703060329053/nvidia-logo-vert.png" alt="NVIDIA" height="100"/>
<img src="https://tdcommons.ai/static/images/logonav.png" alt="tdc" height="80"/>
<img src="https://upload.wikimedia.org/wikipedia/commons/9/96/Pytorch_logo.png" alt="Pytorch" height="80"/>