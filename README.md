## 🏥 Medical Insurance Model using Linear Regression by Manas Gupta

### Project Overview
This project predicts medical insurance costs for individuals based on demographic and lifestyle attributes such as age, gender, BMI, number of children, smoking status, and region. By using Linear Regression, the model estimates expected insurance charges to help analyze cost drivers.

### Dataset Information
* **Source:** Kaggle Medical Cost Personal Datasets[cite: 15]
* **Total Records:** 1,338[cite: 15]
* **Features:**
  * `age`: Age of the primary beneficiary[cite: 15]
  * `sex`: Contractor gender (female, male)[cite: 15]
  * `bmi`: Body mass index[cite: 15]
  * `children`: Number of children/dependents covered[cite: 15]
  * `smoker`: Smoking status (yes, no)[cite: 15]
  * `region`: Beneficiary's residential area in the US (northeast, northwest, southeast, southwest)[cite: 15]
  * `charges`: Individual medical costs billed by health insurance[cite: 15]

### Setup and Execution
1. Ensure Python and required libraries (`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`) are installed.
2. Place the `insurance.csv` file in your root project folder.
3. Run the script using your terminal or environment:
   ```bash
   python Medical_insurance_cost_prediction.py