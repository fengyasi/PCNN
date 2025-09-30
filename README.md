# PCNN
The source code of point cloud neural network for predicting endoleak after EVAR
# EVAR Postoperative Analysis Toolkit

## Dataset

- The raw `.stl` files of vascular models were converted to `.npz` format using `datasets/stl2npz.py`
- The resulting `.npz` files were then normalized using `normalize_npz.py` to generate the final `EVARnpz` dataset
- All vessels are renamed according to their **hospitalization ID**
- For any naming discrepancies, the master Excel file should be considered as the authoritative source
- **Final dataset: 381 cases**

## Project Structure

```
├── datasets/
│   ├── stl2npz.py          # Convert STL files to NPZ format
│   └── normalize_npz.py    # Normalize NPZ data
├── split_5fold+t.py        # 5-fold cross-validation splitting
├── main.py                 # Main training and testing script

```

## Usage

### 1. Data Preparation

```bash
# Convert STL files to NPZ format
python datasets/stl2npz.py

# Normalize the NPZ files
python datasets/normalize_npz.py
```

### 2. Data Splitting

Run the script to perform 5-fold cross-validation split:

```bash
python split_5fold+t.py
```

### 3. Training and Testing

The main script for training and evaluation:

```bash
python main.py
```

**Configuration Notes:**
- Saves only the model with the highest validation AUC
- By default, overwrites previous models in the same save path
- Modify `SAVE_DIR` in the script to change model output directory
- Prints evaluation metrics for each fold and aggregated test results

### 4. Complete Pipeline (Recommended)

Run the entire pipeline with a single command:

```bash
./split_train_test.sh \
  --randomseed [SEED] \    # Random seed for data splitting
  --task [TASK_ID] \       # Task identifier (1-3)
  --lr [LEARNING_RATE] \   # Learning rate (e.g., 1e-3)
  --epochs [EPOCHS]        # Number of training epochs
```


### Training Parameters
```python
# Default values (modifiable in main.py or via command line)
learning_rate = 1e-3
epochs = 100
batch_size = 32
random_seed = 42
```

## Output

- **Models**: Saved in specified `SAVE_DIR` (highest AUC only)
- **Metrics**: Printed to console for each fold and final test performance
- **Logs**: Training progress and validation metrics

## Important Notes

- **Reproducibility**: Use `--randomseed` for consistent data splits
- **Model Management**: Modify `SAVE_DIR` in `main.py` to prevent overwriting
- **Naming Convention**: Always refer to Excel file for correct hospitalization IDs
- **Data Integrity**: 2 cases excluded due to missing STL files

## Dependencies

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Scikit-learn

