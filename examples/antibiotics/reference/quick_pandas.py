import pandas as pd


df = pd.read_csv("examples/antibiotics/synthetic_antibiotics_labels.csv")
df.to_excel("examples/antibiotics/temp.xlsx")
# print(df)
# df['diff'] = pd.to_datetime(df['end_date']) -  pd.to_datetime(df['start_date'])
# df['note_id'] = df['note_id'].str.replace('note_', '', regex=False)
# df.to_excel("examples/antibiotics/synthetic_evaluation_labels.xlsx",
#             columns=['note_id','drug_name','start_date','end_date','diff'],index=False)
