# Chronic Kidney Disease Analysis

This project analyzes chronic kidney disease data using machine learning techniques and visualizations.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the visualization script:
```bash
python violin_plot.py
```

The script will:
1. Load and clean the kidney disease dataset
2. Create violin plots for numerical features
3. Save the plots in the `plots` directory

## Data Description

The dataset (`kidney_disease.csv`) contains various health metrics including:
- age: Patient's age
- bp: Blood pressure
- bgr: Blood glucose random
- bu: Blood urea
- sc: Serum creatinine
- sod: Sodium
- pot: Potassium
- hemo: Hemoglobin
- pcv: Packed cell volume
- wc: White blood cell count
- rc: Red blood cell count

## Output

The violin plots will be saved in the `plots` directory as `kidney_disease_violin_plots.png`. These plots show the distribution of various health metrics grouped by CKD classification.