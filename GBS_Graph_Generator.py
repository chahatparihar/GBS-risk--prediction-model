
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("GBS_Pune_Dataset.csv")

# Set plot style
sns.set(style="whitegrid")

# Plot 1: Area-wise distribution
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Area', order=df['Area'].value_counts().index, palette='viridis')
plt.title('Area-wise GBS Case Distribution in Pune')
plt.xlabel('Area')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("graph_area_wise.png")
plt.close()

# Plot 2: Gender-wise distribution
plt.figure(figsize=(6, 6))
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightblue', 'lightpink'])
plt.title('Gender-wise Distribution')
plt.ylabel('')
plt.tight_layout()
plt.savefig("graph_gender_pie.png")
plt.close()

# Plot 3: Age Distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=10, kde=True, color='skyblue')
plt.title('Age Distribution of GBS Patients')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig("graph_age_distribution.png")
plt.close()

# Plot 4: Recovery Status
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Recovery_Status', order=['Recovered', 'Hospitalized', 'Deceased'], palette='Set2')
plt.title('Recovery Status of Patients')
plt.xlabel('Status')
plt.ylabel('Number of Patients')
plt.tight_layout()
plt.savefig("graph_recovery_status.png")
plt.close()

# Plot 5: Trend of Onset Dates
df['Onset_Date'] = pd.to_datetime(df['Onset_Date'])
onset_trend = df.groupby('Onset_Date').size()
plt.figure(figsize=(10, 5))
onset_trend.plot(kind='line', marker='o', color='orange')
plt.title('Daily New GBS Cases')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.tight_layout()
plt.savefig("graph_onset_trend.png")
plt.close()
