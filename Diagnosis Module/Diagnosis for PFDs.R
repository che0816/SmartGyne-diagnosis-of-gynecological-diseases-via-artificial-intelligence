# Clear environment and set working directory
rm(list = ls())
setwd("path/to/your/project")  # Set relative path to project root

# Load required packages
library(readxl)    # Excel file reading
library(xlsx)      # Legacy Excel writing
library(openxlsx)  # Modern Excel writing
library(caret)     # Machine learning utilities
library(e1071)     # Statistical learning
library(gmodels)   # Goodness-of-fit models
library(stringr)   # String manipulation
library(rpart)     # Recursive partitioning
library(maptree)   # Tree visualization
library(TH.data)   # Data handling
library(rpart.plot)# Tree plotting
library(rattle)    # Fancy tree visualization
library(pROC)      # ROC curve analysis
library(randomForest) # Random forest modeling
library(tidyr)     # Data tidying

# ==============================================================================
# DATA IMPORT AND PREPROCESSING
# ==============================================================================
# Import surgical and normal datasets
surgery <- read_excel("data/surgery_final_accto_symptom_xiu.xlsx")
normal <- read_excel("data/normal.xlsx")
combined_data <- rbind(surgery, normal)  # Merge datasets

# Define variable groups
numeric_vars <- c(
  "孕次", "流产次数", "产次", "难产次数", "分娩最大胎儿体重",
  "绝经(年)", "绝经后HRT治疗(年)",
  "咳嗽时漏尿(年)", "走路时漏尿(年)", "听水声漏尿(年)", "BMI",
  "A(a)", "B(a)", "C", "gh", "Pb", "Tvl", "A(p)", "B(p)", "D",
  "初始排尿欲时膀胱容量(ml)", "常态排尿欲时膀胱容量(ml)", "膀胱最大容量(ml)", 
  "平均排尿率(ml/s)", "残余尿量(ml)", "尿流率峰值流率(ml/s)",
  "尿流率达到峰值流率时间(s)", "尿流率排尿量(ml)", "尿流率流率时间(s)",
  "尿流率排尿时间(s)", "尿流率平均流率(ml/s)", "尿残留(ml)", "排余尿(ml)",
  "结果(g)", "手术时年龄"
)

factor_vars <- setdiff(c(
  "民族", "婚姻状况", "居住地", "文化程度", "职业", "就业状态", "劳动强度",
  "目前月经状态", "绝经方式", "本次住院术前性生活", "吸烟情况", "饮酒情况",
  "尿失禁有无", "下尿路症状有无", "便失禁有无", "生殖道组织物脱出有无",
  "生殖道分泌物异常有无", "逼尿肌不稳定", "子宫大小", "阴性阳性",
  "咳嗽时漏尿", "走路时漏尿", "听水声漏尿"
), numeric_vars)

# Convert variables to appropriate types
convert_to_numeric <- function(var) as.numeric(as.character(combined_data[[var]]))
convert_to_factor <- function(var) as.factor(combined_data[[var]])

combined_data[, numeric_vars] <- lapply(numeric_vars, convert_to_numeric)
combined_data[, factor_vars] <- lapply(factor_vars, convert_to_factor)

# Clean column names (replace special characters)
clean_colnames <- function(x) {
  x <- gsub("[()/]", "_", x)
  make.names(x, unique = TRUE)
}
colnames(combined_data) <- clean_colnames(colnames(combined_data))

# ==============================================================================
# VARIABLE SELECTION FUNCTION
# ==============================================================================
variable_selection <- function(model_type, stage) {
  # """
  # Select variables based on model type and analysis stage.
  # Args:
  #   model_type (char): Model identifier (e.g., "model5", "model2")
  #   stage (char): Analysis stage (e.g., "stage2", "stage3_UI")
  # Returns:
  #   vector: Selected variable names
  # """
  if (model_type == "model5") {
    return(c(
      "产次", "BMI", "尿失禁有无", "生殖道组织物脱出有无",
      "A_a_", "B_a_", "C", "gh", "Pb", "Tvl", "A_p_", "B_p_", "D",
      "初始排尿欲时膀胱容量_ml_", "常态排尿欲时膀胱容量_ml_", "膀胱最大容量_ml_",
      "平均排尿率_ml_s_", "残余尿量_ml_", "逼尿肌不稳定",
      "尿流率峰值流率_ml_s_", "尿流率达到峰值流率时间_s_", "尿流率排尿量_ml_",
      "尿流率流率时间_s_", "尿流率排尿时间_s_", "尿流率平均流率_ml_s_",
      "尿残留_ml_", "排余尿_ml_", "子宫大小", "结果_g_", "手术时年龄",
      "咳嗽时漏尿", "走路时漏尿", "听水声漏尿"
    ))
  } else {
    # Define variable groups for different models/stages
    pfd_vars <- c("手术时年龄", "BMI", "产次", "生殖道组织物脱出有无", "尿失禁有无")
    ui_vars <- c(
      "初始排尿欲时膀胱容量_ml_", "常态排尿欲时膀胱容量_ml_", "膀胱最大容量_ml_",
      "平均排尿率_ml_s_", "残余尿量_ml_", "逼尿肌不稳定",
      "尿流率峰值流率_ml_s_", "尿流率达到峰值流率时间_s_", "尿流率排尿量_ml_",
      "尿流率流率时间_s_", "尿流率排尿时间_s_", "尿流率平均流率_ml_s_",
      "尿残留_ml_", "排余尿_ml_", "结果_g_", "咳嗽时漏尿", "走路时漏尿", "听水声漏尿"
    )
    pop_vars <- c("A_a_", "B_a_", "C", "gh", "Pb", "Tvl", "A_p_", "B_p_", "D", "子宫大小")
    
    if (model_type == "model2") {
      stage_mapping <- list(
        stage2 = pfd_vars,
        stage3_UI = ui_vars,
        stage3_POP_f = pop_vars,
        stage3_POP_m = pop_vars,
        stage3_POP_b = pop_vars
      )
      return(stage_mapping[[stage]])
    } else {
      return(unique(c(pfd_vars, ui_vars, pop_vars)))
    }
  }
}

# ==============================================================================
# DATA CLEANING AND MODELING PIPELINE
# ==============================================================================
set.seed(2020)  # Ensure reproducibility
analysis_stages <- c("stage2", "stage3_UI", "stage3_POP_f", "stage3_POP_m", "stage3_POP_b")
results <- list()

for (current_stage in analysis_stages) {
  # Subset data for the current stage
  stage_data <- combined_data[!is.na(combined_data[[current_stage]]), ]
  
  # Further filter based on stage logic
  if (current_stage == "stage3_UI") {
    stage_data <- stage_data[stage_data$stage2 == "UI,POP", ]
  } else if (grepl("stage3_POP", current_stage)) {
    stage_data <- stage_data[stage_data$生殖道组织物脱出有无 == "有", ]
  }
  
  # Select variables for modeling
  selected_vars <- variable_selection(model_type = "model2", stage = current_stage)
  model_data <- stage_data[, c(current_stage, selected_vars)]
  
  # Handle missing values (example: remove rows with missing values)
  model_data <- na.omit(model_data)
  
  # Convert target variable to factor
  model_data[[current_stage]] <- as.factor(model_data[[current_stage]])
  
  # Initialize performance metrics
  accuracy_scores <- numeric(1000)
  
  for (iteration in 1:1000) {
    # Split data into training and test sets (75%/25%)
    split_index <- createDataPartition(
      model_data[[current_stage]],
      times = 1,
      p = 0.75,
      list = FALSE
    )
    train_set <- model_data[split_index, ]
    test_set <- model_data[-split_index, ]
    
    # Build CART model
    model_formula <- as.formula(paste(current_stage, "~ ."))
    cart_model <- rpart(model_formula, method = "class", data = train_set)
    
    # Prune the tree (for non-UI stages)
    if (grepl("stage3_POP", current_stage)) {
      best_cp <- cart_model$cptable[which.min(cart_model$cptable[,"xerror"]), "CP"]
      cart_model <- prune(cart_model, cp = best_cp)
    }
    
    # Make predictions
    predictions <- predict(cart_model, newdata = test_set[, -1], type = "prob")
    class_pred <- colnames(predictions)[apply(predictions, 1, which.max)]
    
    # Calculate accuracy
    confusion_matrix <- table(
      Actual = test_set[[current_stage]],
      Predicted = class_pred
    )
    accuracy_scores[iteration] <- sum(diag(confusion_matrix)) / nrow(test_set)
    
    # Save plot for the first iteration
    if (iteration == 1) {
      png(
        filename = file.path("results", paste0(current_stage, "_CART.png")),
        width = 1500,
        height = 900
      )
      plot(cart_model, main = current_stage)
      text(cart_model, use.n = TRUE)
      dev.off()
    }
  }
  
  # Aggregate performance metrics
  results[[current_stage]] <- list(
    accuracy_mean = mean(accuracy_scores),
    accuracy_std = sd(accuracy_scores),
    model = cart_model
  )
}

# ==============================================================================
# SAVE RESULTS TO EXCEL
# ==============================================================================
# Create results directory if not exists
if (!dir.exists("results")) dir.create("results")

# Write performance metrics to Excel
write.xlsx2(
  x = do.call(rbind, lapply(results, function(x) x$accuracy_mean)),
  file = file.path("results", "model2_performance.xlsx"),
  sheetName = "Accuracy",
  row.names = TRUE
)

cat("Modeling completed. Results saved to 'results/' directory.\n")