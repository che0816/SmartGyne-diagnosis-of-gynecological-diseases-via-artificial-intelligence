# Clear environment and set working directory
rm(list = ls())
setwd("path/to/your/project")  # Set relative path to project root

# Load required packages (sorted alphabetically for clarity)
library(Matrix)
library(jsonlite)
library(pROC)
library(readxl)
library(randomForest)
library(rattle)
library(rpart)
library(rpart.plot)
library(xlsx)

# Set Chinese font for plot rendering (Windows-specific)
# Note: Requires SimHei font installed on the system
windowsFonts(myFont = windowsFont("SimHei"))
par(family = "myFont")

# ==============================================================================
# DATA PREPROCESSING FUNCTION
# ==============================================================================
preprocess_data <- function(data_dir = "data/", result_dir = "results/") {
  # """
  # Perform data preprocessing for medical diagnosis modeling.
  # Args:
  #   data_dir: Path to data directory (default: "data/")
  #   result_dir: Path to results directory (default: "results/")
  # Returns:
  #   List containing preprocessed data matrices and metadata
  # """
  cat("Starting data preprocessing...\n")
  
  # Load raw data and sparse matrices
  load(file.path(data_dir, "combine_standard_final_notseg_factor.RData"))
  
  # Read feature list and sparse matrix files
  feature_list <- jsonlite::stream_in(
    file = file.path(data_dir, "sparse_matrix/feature_table0109/column_list.json")
  )
  zhusu_sm <- Matrix::readMM(
    file = file.path(data_dir, "sparse_matrix/feature_table0109/sparse_matrix_cc_notseg.mtx")
  )
  sm0 <- Matrix::readMM(
    file = file.path(data_dir, "sparse_matrix/data_final/sparse_matrix_cc_notseg.mtx")
  )
  
  # Label columns and convert disease IDs to characters
  colnames(zhusu_sm) <- paste0("Chief complaint-", feature_list[1, ])
  disease_list[1, ] <- as.character(disease_list[1, ])
  
  # Merge matrices and adjust indices
  zhusu_sm[, c(1, 2)] <- sm0[, c(1, 2)]
  zhusu_sm[, c(1, 2)] <- zhusu_sm[, c(1, 2)] + 1  # Adjust for 1-based indexing
  
  # Combine multiple data segments (e.g., medical history, examination results)
  data_segments <- c("Chief complaint", "Past history", "Present illness", 
                     "Physical examination", "Auxiliary examination", "Treatment advice")
  for (i in 2:5) {  # Exclude first segment (already processed)
    sm <- Matrix::readMM(
      file = file.path(data_dir, "sparse_matrix/feature_table0109/sparse_matrix_", 
                       data_segments[i], "_notseg.mtx")
    )
    colnames(sm) <- paste0(data_segments[i], "-", feature_list[1, ])
    sm <- sm[, -c(1, 2)]  # Remove redundant ID columns
    zhusu_sm <- cbind(zhusu_sm, sm)
  }
  
  # Analyze disease frequency distribution
  disease_freq <- as.data.frame(table(zhusu_sm[, 2]))
  disease_freq <- disease_freq[order(-disease_freq$Freq), ]
  disease_freq$disease <- as.character(disease_list[1, ])[
    as.numeric(as.character(disease_freq$Var1))
  ]
  write.csv(disease_freq, 
            file.path(result_dir, "diagnosis_count_frompan_final0109_notseg.csv"),
            fileEncoding = "GBK", row.names = FALSE)
  
  # Load and process full dataset
  fulldata <- read_excel(file.path(data_dir, "full_data_whole.xlsx"), skip = 1)
  fulldata$`Chief complaint-ID` <- fulldata$`Chief complaint-ID` + 1  # Adjust ID indexing
  
  # Save processed data
  save(zhusu, disease_list, fulldata, 
       file = file.path(data_dir, "combine_final_notseg0109.RData"))
  
  cat("Data preprocessing completed.\n")
  return(list(zhusu = zhusu, disease_list = disease_list, fulldata = fulldata))
}

# ==============================================================================
# CROSS-VALIDATION FUNCTION
# ==============================================================================
CVgroup <- function(k_folds, data_size, seed = 2022) {
  # """
  # Generate cross-validation fold indices.
  # Args:
  #   k_folds: Number of folds
  #   data_size: Total number of samples
  #   seed: Random seed for reproducibility (default: 2022)
  # Returns:
  #   List of fold indices
  # """
  set.seed(seed)
  fold_labels <- rep(1:k_folds, ceiling(data_size/k_folds))[1:data_size]
  shuffled_labels <- sample(fold_labels, data_size)
  return(lapply(1:k_folds, function(x) which(shuffled_labels == x)))
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
cal <- function(x) {
  # """Calculate mean and round to 3 decimal places"""
  round(mean(x, na.rm = TRUE), 3)
}

process_location <- function(str) {
  # """Extract location information from string (e.g., parentheses)"""
  if (grepl("\\（", str)) {
    str <- unlist(strsplit(str, "（"))[2]
    str <- unlist(strsplit(str, "）"))[1]
    str <- unlist(strsplit(str, "，"))
  }
  return(str)
}

# ==============================================================================
# MODEL BUILDING AND EVALUATION FUNCTION
# ==============================================================================
build_and_evaluate_model <- function(data, tree_structures, model_type = 1, aggregate = 1) {
  # """
  # Build and evaluate hierarchical decision tree models.
  # Args:
  #   data: Output from preprocess_data()
  #   tree_structures: List of tree hierarchy configurations
  #   model_type: 1 = CART, 2 = BERT, 3 = Ensemble (default: 1)
  #   aggregate: Whether to aggregate predictions (default: 1)
  # """
  cat("Starting model building and evaluation...\n")
  
  zhusu <- data$zhusu
  disease_list <- data$disease_list
  fulldata <- data$fulldata
  
  # Split data into training and test sets
  # Note: 'mat_model' must be defined in the global environment (potential improvement needed)
  ind <- caret::createDataPartition(
    as.factor(mat_model[,"disease.tree.id"]),
    times = 1, p = 0.7, list = FALSE
  )
  train_ids <- mat_model[ind, "Chief complaint-ID"]
  test_ids <- mat_model[-ind, "Chief complaint-ID"]
  
  # Filter data by IDs
  zhusu_filtered <- zhusu[which(zhusu[, 1] %in% c(train_ids, test_ids)), ]
  
  # Process each hierarchical tree structure
  for (m in 1:length(tree_structures)) {
    cat(paste("Processing tree structure", m, "...\n"))
    
    # Read tree structure from Excel
    tree <- openxlsx::read.xlsx(
      file.path("data/", "tree_table0124.xlsx"), 
      sheet = 1, startRow = 1
    )
    tree[3, 4] <- unlist(strsplit(tree[3, 4], "（"))[1]  # Clean category names
    tree[4, 4] <- unlist(strsplit(tree[4, 4], "（"))[1]
    
    # Create disease ID mapping
    disease_map <- data.frame(disease_tree_id = NA, disease_name = NA)
    for (jj in 1:(ncol(tree) - 1)) {
      for (ii in 1:nrow(tree)) {
        if (!is.na(tree[ii, jj])) {
          if (grepl("\\（", tree[ii, jj])) {
            disease_map <- rbind(disease_map, data.frame(
              disease_tree_id = as.numeric(paste0(m, ii, "00", jj)),
              disease_name = unlist(strsplit(tree[ii, jj], "（"))[1]
            ))
          } else {
            disease_map <- rbind(disease_map, data.frame(
              disease_tree_id = as.numeric(paste0(m, ii, "00", jj)),
              disease_name = tree[ii, jj]
            ))
          }
        }
      }
    }
    disease_map <- disease_map[-1, ]  # Remove initial NA row
    
    # Process tree hierarchy levels
    tree$`1` <- lapply(tree$`1`, process_location)
    tree$`2` <- lapply(tree$`2`, process_location)
    tree$`3` <- lapply(tree$`3`, process_location)
    tree$`4` <- lapply(tree$`4`, process_location)
    tree$diseases <- lapply(tree$diseases, function(str) unlist(strsplit(str, ";")))
    
    # Initialize result matrices
    stop_condition <- as.data.frame(matrix(0, 
                                           ncol = length(sort(tree_structures[[m]]$hierarchy)) + 1))
    zhusu_matrix <- cbind(zhusu_filtered[, 1], zhusu_filtered)
    colnames(zhusu_matrix)[1] <- "disease_tree_id"
    zhusu_matrix[, "disease_tree_id"] <- NA
    
    # Map diseases to hierarchical tree IDs
    for (aa in 1:nrow(tree)) {
      if (!is.na(tree$diseases[aa])) {
        match_indices <- which(disease_list[1, zhusu_matrix[, "Chief complaint-diagnosis"]] %in% 
                                 unlist(tree$diseases[aa]))
        if (length(match_indices) > 0) {
          patient_ids <- zhusu_matrix[match_indices, "Chief complaint-ID"]
          zhusu_matrix[which(zhusu_matrix[, "Chief complaint-ID"] %in% patient_ids), 
                       "disease_tree_id"] <- as.numeric(paste0(m, aa, "00", 4))
        }
      }
    }
    
    # Iterate over hierarchy levels
    for (j in sort(tree_structures[[m]]$hierarchy)) {
      cat(paste("Processing hierarchy level", j, "...\n"))
      
      current_level <- tree_structures[[m]]$hang[[paste0("hierarchy", j)]]
      performance_metrics <- list()
      
      for (i in 1:length(current_level)) {
        category_id <- paste0(m, current_level[[i]], "00", j)
        category_data <- zhusu_matrix[which(zhusu_matrix[, "disease_tree_id"] %in% category_id), ]
        
        if (nrow(category_data) > 50) {  # Filter underrepresented categories
          # Feature selection based on column sums
          col_sums <- apply(category_data, 2, sum)
          cutoff <- nrow(category_data) * 0.01
          model_data <- category_data[, col_sums > cutoff]
          
          # Convert to data frame and factorize
          model_df <- as.data.frame(as.matrix(model_data))
          model_df[, "disease_tree_id"] <- factor(
            model_df[, "disease_tree_id"],
            labels = disease_map$disease_name[match(model_df$disease_tree_id, disease_map$disease_tree_id)]
          )
          model_df[, -c(1:3)] <- lapply(model_df[, -c(1:3)], as.factor)
          
          # Split into train/test
          train_subset <- model_df[model_df$`Chief complaint-ID` %in% train_ids, ]
          test_subset <- model_df[model_df$`Chief complaint-ID` %in% test_ids, ]
          
          if (nrow(test_subset) == 0) {  # Handle cases with no test samples
            performance_metrics[[i]] <- list(accuracy = 0, sensitivity = 0, specificity = 0)
            next
          }
          
          # Build CART model
          cart_model <- rpart(
            disease_tree_id ~ ., 
            method = "class", 
            data = train_subset[, -c(2, 3)]  # Exclude ID columns
          )
          pruned_model <- prune(cart_model, 
                                cp = cart_model$cptable[which.min(cart_model$cptable[,"xerror"]), "CP"])
          
          # Predict and evaluate
          predictions <- predict(pruned_model, test_subset[, -c(1:3)], type = "prob")
          class_predictions <- colnames(predictions)[apply(predictions, 1, which.max)]
          confusion_mat <- table(
            Actual = test_subset$disease_tree_id, 
            Predicted = class_predictions
          )
          
          # Calculate metrics
          accuracy <- sum(diag(confusion_mat)) / sum(confusion_mat)
          sensitivity <- diag(confusion_mat) / rowSums(confusion_mat)
          specificity <- apply(confusion_mat, 1, function(row) sum(row[-which.max(row)])/sum(row))
          
          performance_metrics[[i]] <- list(
            accuracy = accuracy,
            sensitivity = sensitivity,
            specificity = specificity,
            confusion_matrix = confusion_mat
          )
          
          # Save plots and results (example)
          png(file.path("results/cart_figures/", paste0("hierarchy", j, "_", i, ".png")), 
              width = 1800, height = 900)
          par(mfrow = c(1, 2))
          plot(pruned_model, main = "Pruned Tree")
          text(pruned_model, use.n = TRUE)
          plotcp(pruned_model, main = "Cross-Validation Error")
          dev.off()
        }
      }
      
      # Aggregate metrics across categories
      aggregated_metrics <- do.call(rbind, performance_metrics)
      write.csv(aggregated_metrics, 
                file.path("results/", paste0("hierarchy_", j, "_metrics.csv")),
                row.names = FALSE)
    }
  }
  
  cat("Model building and evaluation completed.\n")
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
main <- function() {
  # """Main execution pipeline"""
  # Parameters
  MODEL_TYPE <- 1  # 1 = CART, 2 = BERT, 3 = Ensemble
  AGGREGATE_PREDICTIONS <- 1  # 1 = Enable aggregation
  
  # Preprocess data
  processed_data <- preprocess_data()
  
  # Define hierarchical tree structure
  tree_structures <- list(
    list(
      tree_id = 1,
      hierarchy = c(2, 3, 4),
      hang = list(
        hierarchy2 = list("Female reproductive system diseases" = 1:24),
        hierarchy3 = list("Non-neoplastic" = 1:12, "Neoplastic" = 13:24),
        hierarchy4 = list(
          "Inflammatory" = 1:5,
          "Non-inflammatory" = 6:12,
          "Benign tumors" = 13:15,
          "Malignant tumors" = 16:21,
          "Uncertain tumors" = 22:24
        )
      )
    )
  )
  
  # Execute modeling
  build_and_evaluate_model(
    data = processed_data,
    tree_structures = tree_structures,
    model_type = MODEL_TYPE,
    aggregate = AGGREGATE_PREDICTIONS
  )
  
  cat("Program execution completed.\n")
}

# Run main function
main()