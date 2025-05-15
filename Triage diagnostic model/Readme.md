# Triage diagnosis of gynecological diseases


## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Key Features](#key-features)  
3. [Prerequisites](#prerequisites)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [License](#license)  


## Project Overview  
This repository contains the source code for our study on automated diagnosis prediction using medical records and hierarchical decision tree models. The code includes data preprocessing, hierarchical model construction, and validation for medical diagnosis tasks.  


## Key Features  
- **Data Preprocessing**: Standardization of medical diagnoses, sparse matrix construction from electronic health records (EHRs).  
- **Hierarchical Modeling**: Decision tree (CART) models with multi-level disease hierarchies (e.g., female reproductive system diseases → non-neoplastic vs. neoplastic).  
- **Evaluation**: confusion matrices, and performance metrics (accuracy, sensitivity, specificity, F1-score).  
- **Integration**: Support for combining CART with BERT-based embeddings (optional, controlled by `button` parameter).  


## Prerequisites  
### Software Dependencies  
- **R Environment**: Version ≥ 3.6.0  
- **R Packages**:  
  ```r  
  readxl, rpart, rpart.plot, pROC, randomForest, Matrix, jsonlite, xlsx  
  ```  
- **Tools**:  
  - GitHub Desktop (for version control)  
  - RStudio (recommended for development)  

### Data Requirements  
- Input data files (e.g., `combine_standard_final_notseg_factor.RData`, Excel annotation files) must be structured in the `data/` directory as specified in the code.  


## Installation  
**Install R Packages**:  
   ```r  
   install.packages(c("readxl", "rpart", "pROC", "Matrix", "jsonlite", "xlsx"))  
   ```  


## Usage  
### Run the Analysis  
1. **Set Configuration**:  
   - Adjust parameters in `Triage for gynecological diseases.R` (e.g., `button` for model type, `aggregated` for ensemble methods).  
   - Update file paths in `preprocess_data()` and `build_and_evaluate_model()` if your data structure differs.  

2. **Execute the Script**:  
   ```r  
   source("src/Triage for gynecological diseases.R")  
   ```  

### Expected Outputs  
- Decision tree plots in `results/cart_figures/`  
- Evaluation metrics in Excel files under `results/`  
- Confusion matrices and prediction error logs in `results/`  


## License  
This project is licensed under the **MIT License**.  


### Notes  
- The code is provided as-is for research purposes. For production use, additional validation and optimization may be required.  
