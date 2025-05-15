
# SmartGyne: Code Repository for Gynecological Disease Diagnosis System  

This repository contains the source code for the SmartGyne system, a comprehensive AI-driven framework for gynecological disease diagnosis. The code is structured into four main modules, corresponding to different stages of the diagnostic pipeline. It is designed to be used alongside the accompanying research paper, which details the methodology and evaluation of the system.  


## Table of Contents  
1. [TopWORDS_Series](#topwords_series)  
2. [Knowledge Graph](#knowledgegraph)  
3. [Triage Module](#triage-diagnostic-model)  
4. [Diagnosis Module](#accurate-diagnostic-model)  


## <a name="topwords_series"></a>1. TopWORDS_Series  
### Description  
This module implements the TopWORDS series of algorithms for natural language processing (NLP) on electronic medical records (EMRs). The algorithms simultaneously perform **high-quality word discovery** and **text segmentation**, enabling the recognition of various medical terms in the EMR database. The output is a structured feature table, which encodes the presence/absence of thousands of medical features by converting semi-structured EMRs into a machine-readable format.  


## <a name="knowledgegraph"></a>2. Knowledge Graph  
### Description  
This folder contains code for constructing a knowledge graph that integrates domain knowledge for gynecological diagnosis. The knowledge graph serves as a semantic foundation for the diagnostic models, encoding relationships between diseases, symptoms, risk factors, and medical guidelines.  


## <a name="triage-diagnostic-model"></a>3. Triage Module  
### Description  
This module implements the triage module, designed to provide initial disease categorization for gynecological patients. The triage module uses a hierarchical architecture where each internal node of the disease hierarchy contains a local disease classifier. The model performs sequential triage diagnosis by traversing the disease hierarchy, narrowing down potential conditions at each level.  


## <a name="accurate-diagnostic-model"></a>4. Diagnosis Module  
### Description  
This folder contains the code for the diagnosis module for pelvic floor disorders, based on the CART (Classification and Regression Trees) algorithm. The diagnosis module is attached to the leaf nodes of the disease hierarchy to provide precise diagnoses. It is trained and optimized using specialist case data under the guidance of the knowledge graph, ensuring high accuracy for rare or complex conditions.  
