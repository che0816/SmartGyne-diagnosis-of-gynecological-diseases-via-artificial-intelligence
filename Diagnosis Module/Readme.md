# Diagnosis for Pelvic Floor Disorders (PFDs) 
**Repository for statistical modeling of pelvic floor disorders using clinical data**  


## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Requirements](#requirements)   
3. [Usage Instructions](#usage-instructions)  
4. [Code Overview](#code-overview)  
5. [License](#license)  


## 1. Project Overview  
This repository contains the R code for a study on pelvic floor disorder (PFD) modeling, including data preprocessing, variable selection, and machine learning analysis (CART). The code is designed to reproduce the analyses described in the accompanying research paper.  

**Key Features:**  
- Data preprocessing for clinical datasets (Excel files).  
- Hierarchical variable selection for different PFD stages (UI = urinary incontinence, POP = pelvic organ prolapse).  
- Cross-validated CART modeling with tree pruning and performance evaluation.  
- Generation of confusion matrices, accuracy metrics, and decision tree visualizations.  


## 2. Requirements  
### Software Dependencies  
- **R** (version ≥ 3.6.0)  
- **Required Packages**:  
  ```r
  readxl, xlsx, openxlsx, caret, e1071, gmodels, stringr, 
  rpart, maptree, TH.data, rpart.plot, rattle, pROC, randomForest, tidyr
  ```  
  Install dependencies with:  
  ```r
  install.packages(c("readxl", "caret", "rpart", "pROC", "randomForest"))
  ```


## 3. Usage Instructions  
### Run the Analysis  
Open `src/Diagnosis for PFDs.R` in RStudio and execute the script. The code will:  
1. Preprocess data (convert types, clean column names).  
2. Perform variable selection for each PFD stage.  
3. Train and evaluate CART models.  
4. Save decision tree plots to `results/` and accuracy metrics to `model2_performance.xlsx`.  


## 4. Code Overview  
### Key Functions  
1. **Data Preprocessing**:  
   - `convert_to_numeric()` and `convert_to_factor()` handle variable type conversions.  
   - `clean_colnames()` removes special characters from column names (e.g., `"(ml)"` → `_ml_`).  

2. **Variable Selection**:  
   - `variable_selection(model_type, stage)` selects features based on the model (e.g., "model2") and analysis stage (e.g., "stage3_POP_f").  

3. **Modeling Pipeline**:  
   - Splits data into training/test sets using `caret::createDataPartition`.  
   - Builds and prunes CART trees with `rpart::rpart` and `rpart::prune`.  
   - Evaluates performance via confusion matrices and accuracy scores.  

### Outputs  
- **Plots**: Decision tree visualizations for each stage (e.g., `stage3_UI_CART.png`).  
- **Metrics**: Mean accuracy and standard deviation saved in `model2_performance.xlsx`.  


## 5. License  
This project is licensed under the **MIT License**.
