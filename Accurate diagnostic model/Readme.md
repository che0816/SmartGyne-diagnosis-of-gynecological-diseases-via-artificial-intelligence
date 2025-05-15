# Pelvic Floor Disorder (PFD) Modeling Code  
**Repository for statistical modeling of pelvic floor disorders using clinical data**  


## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Requirements](#requirements)  
3. [Repository Structure](#repository-structure)  
4. [Usage Instructions](#usage-instructions)  
5. [Code Overview](#code-overview)  
6. [Citation](#citation)  
7. [License](#license)  


## 1. Project Overview  
This repository contains the R code for a study on pelvic floor disorder (PFD) modeling, including data preprocessing, variable selection, and machine learning analysis (CART decision trees). The code is designed to reproduce the analyses described in the accompanying research paper.  

**Key Features:**  
- Data preprocessing for clinical datasets (Excel files).  
- Hierarchical variable selection for different PFD stages (UI = urinary incontinence, POP = pelvic organ prolapse).  
- Cross-validated CART modeling with tree pruning and performance evaluation.  
- Generation of confusion matrices, accuracy metrics, and decision tree visualizations.  


## 2. Requirements  
### Software Dependencies  
- **R** (version ≥ 4.0.0)  
- **Required Packages**:  
  ```r
  readxl, xlsx, openxlsx, caret, e1071, gmodels, stringr, 
  rpart, maptree, TH.data, rpart.plot, rattle, pROC, randomForest, tidyr
  ```  
  Install dependencies with:  
  ```r
  install.packages(c("readxl", "caret", "rpart", "pROC", "randomForest"))
  ```

### Data Format  
- Input data files (`surgery_final_accto_symptom_xiu.xlsx` and `normal.xlsx`) should be placed in a `data/` directory.  
- Columns include clinical variables (e.g., "产次" = parity, "BMI"), diagnostic outcomes (e.g., "尿失禁有无" = urinary incontinence status), and staging variables (e.g., "stage3_UI").  


## 3. Repository Structure  
```
your-repo/
├─ data/                # Input data files (anonymized clinical data)
│  ├─ surgery_final_accto_symptom_xiu.xlsx
│  └─ normal.xlsx
├─ results/             # Auto-generated outputs (plots, metrics)
│  ├─ stage2_CART.png    # Example decision tree plot
│  └─ model2_performance.xlsx  # Accuracy metrics
├─ src/                 # Source code
│  └─ main_analysis.R    # Main analysis script
├─ figures/             # Static figures (optional)
├─ .gitignore           # File exclusion rules
├─ LICENSE              # License file
└─ README.md            # This documentation
```  


## 4. Usage Instructions  
### Step 1: Clone the Repository  
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### Step 2: Prepare Data  
1. Place your anonymized clinical data files in the `data/` directory, maintaining the original filenames.  
2. Ensure column names match those specified in the code (e.g., "孕次", "BMI").  

### Step 3: Run the Analysis  
Open `src/main_analysis.R` in RStudio and execute the script. The code will:  
1. Preprocess data (convert types, clean column names).  
2. Perform variable selection for each PFD stage.  
3. Train and evaluate CART models across 1000 iterations.  
4. Save decision tree plots to `results/` and accuracy metrics to `model2_performance.xlsx`.  


## 5. Code Overview  
### Key Functions  
1. **Data Preprocessing**:  
   - `convert_to_numeric()` and `convert_to_factor()` handle variable type conversions.  
   - `clean_colnames()` removes special characters from column names (e.g., `"(ml)"` → `_ml_`).  

2. **Variable Selection**:  
   - `variable_selection(model_type, stage)` selects features based on the model (e.g., "model2") and analysis stage (e.g., "stage3_POP_f").  

3. **Modeling Pipeline**:  
   - Splits data into training/test sets (75%/25%) using `caret::createDataPartition`.  
   - Builds and prunes CART trees with `rpart::rpart` and `rpart::prune`.  
   - Evaluates performance via confusion matrices and accuracy scores across 1000 iterations.  

### Outputs  
- **Plots**: Decision tree visualizations for each stage (e.g., `stage3_UI_CART.png`).  
- **Metrics**: Mean accuracy and standard deviation saved in `model2_performance.xlsx`.  


## 6. Citation  
If you use this code in your research, please cite the accompanying publication:  
```bibtex
@article{YourName2023,
  title={Pelvic Floor Disorder Modeling Code},
  author={Your Name et al.},
  journal={Journal Name},
  year={2023},
  doi={10.XXXX/XXXX}
}
```  


## 7. License  
This project is licensed under the **MIT License**. See `LICENSE` for details.  

---  

For questions or contributions, contact: your.email@institution.edu  
Last updated: `r format(Sys.Date(), "%B %d, %Y")`
