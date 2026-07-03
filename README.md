# Exact Monotonic Decision Trees (EMDT)

This repository contains the public code artifact and data experiment runners for the paper **Exact Monotonic Decision Trees for Governance-Oriented Classification**.

## Overview
EMDT is a CP-SAT formulation for learning fixed-depth axis-aligned decision trees that are guaranteed to be monotonic with respect to a specified subset of features. 

## Structure
- `experiment/`: Contains all Python scripts to run the EMDT solver and baseline methods across synthetic and real-world datasets (COMPAS, German Credit, Adult Income, Bank Marketing).
- `figures/`: Contains visualizations of decision boundaries and scalability error bars.

## Dependencies
- `ortools`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `pandas`
- `numpy`
