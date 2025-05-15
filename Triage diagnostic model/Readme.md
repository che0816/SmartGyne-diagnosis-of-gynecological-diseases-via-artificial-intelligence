
Here's a suggested structure and content for your GitHub repository's `README.md` file in English, tailored for your research code:


# Title of Your Research Project  
*(e.g., "Medical Text Mining and Decision Tree Modeling for Pelvic Floor Disorder Diagnosis")*  


## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)  
3. [Key Features](#key-features)  
4. [Prerequisites](#prerequisites)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Citation](#citation)  
8. [License](#license)  


## Project Overview  
This repository contains the source code for our study on **[briefly describe your research goal, e.g., "automated diagnosis prediction using medical records and hierarchical decision tree models"**]. The code includes data preprocessing, hierarchical model construction, and cross-validation for medical diagnosis tasks.  


## Repository Structure  
```
.  
├── data/                # Input data and intermediate files  
│   ├── sparse_matrix/   # Sparse matrix files for medical features  
│   ├── RData/           # Saved R data objects (.RData)  
│   └── excel/           # Excel files for annotations and results  
├── results/             # Output results (tables, figures, metrics)  
│   ├── cart_figures/    # Decision tree plots  
│   └── diagnosis_metrics/ # Evaluation metrics  
├── src/                 # Source code  
│   ├── functions/       # Custom functions  
│   ├── main.R           # Main execution script  
│   └── utils.R          # Utility scripts (e.g., data loading)  
├── figures/             # Visualizations (generated or static)  
├── LICENSE              # License file  
└── README.md            # This documentation  
```


## Key Features  
- **Data Preprocessing**: Standardization of medical diagnoses, sparse matrix construction from electronic health records (EHRs).  
- **Hierarchical Modeling**: Decision tree (CART) models with multi-level disease hierarchies (e.g., female reproductive system diseases → non-neoplastic vs. neoplastic).  
- **Evaluation**: Cross-validation, confusion matrices, and performance metrics (accuracy, sensitivity, specificity, F1-score).  
- **Integration**: Support for combining CART with BERT-based embeddings (optional, controlled by `button` parameter).  


## Prerequisites  
### Software Dependencies  
- **R Environment**: Version ≥ 4.0  
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
1. **Clone the Repository**:  
   ```bash  
   git clone https://github.com/your-username/your-repo-name.git  
   cd your-repo-name  
   ```  

2. **Install R Packages**:  
   ```r  
   install.packages(c("readxl", "rpart", "pROC", "Matrix", "jsonlite", "xlsx"))  
   ```  


## Usage  
### Run the Analysis  
1. **Set Configuration**:  
   - Adjust parameters in `main.R` (e.g., `button` for model type, `aggregated` for ensemble methods).  
   - Update file paths in `preprocess_data()` and `build_and_evaluate_model()` if your data structure differs.  

2. **Execute the Script**:  
   ```r  
   source("src/main.R")  
   ```  

### Expected Outputs  
- Decision tree plots in `results/cart_figures/`  
- Evaluation metrics in Excel files under `results/`  
- Confusion matrices and prediction error logs in `results/`  


## Citation  
If you use this code in your research, please cite our paper:  
```bibtex  
@article{YourPaperCitation,  
  title={Your Paper Title},  
  author={Your Name et al.},  
  journal={Journal Name},  
  year={2023},  
  doi={10.1000/your-doi}  
}  
```  


## License  
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.  


## Contact  
For questions or contributions, contact:  
- Your Name: `your.email@example.com`  
- Repository: [https://github.com/your-username/your-repo-name](https://github.com/your-username/your-repo-name)  


### Notes  
- Ensure all data files are properly anonymized before sharing, following ethical and legal standards.  
- The code is provided as-is for research purposes. For production use, additional validation and optimization may be required.  

 

This structure ensures clarity for reviewers and collaborators, while adhering to best practices for open research code. Adjust the details (e.g., citations, dependencies) to match your specific project.
