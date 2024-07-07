import pandas as pd
import matplotlib.pyplot as plt

patients_with_biomarker = pd.read_csv("data/processed/patient_with_biomarkers.csv")

# Count patients per cancer type
patient_counts = patients_with_biomarker['CANCER_TYPE'].value_counts()

# Select top 30 cancer types
top_30 = patient_counts.head(30)

# Plotting with matplotlib
plt.figure(figsize=(3.54,3.54), dpi=600)
top_30.plot(kind='bar', color='C0')
#plt.title('Top 30 Cancer Types by Number of Patients')
plt.xlabel('Cancer Type', fontsize=7)
plt.ylabel('Number of Patients', fontsize=7)
plt.xticks(rotation=45, ha='right', fontsize=5)
plt.yticks(fontsize=5)
plt.tight_layout()

plt.savefig("figures/aacr_patient_cancer_distribution.png")
