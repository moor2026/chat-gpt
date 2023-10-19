import openai
import backoff
import numpy as np
import pandas as pd
import json

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def query_chatgpt(prompts):
  response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=prompts,
      max_tokens=200,
      n=1,
      stop=None,
      temperature=0.1
  )
  jsonStr =response.choices[0].message["content"]

  return json.loads(jsonStr)

openai.api_key = "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"
prompt = [{"role": "system", "content": "You are a veterinarian"},
            {"role": "system", "content": "Answer in JSON format with the main Diagnosis, the Certainty of that diagnosis with values [confirmed, suspected/probable,"
                                          " differential, negated/ruled out, not applicable], the Severity with [mild, moderate, severe, unspecified, not applicable],"
                                          " and the Concurrency with values [active problem, historical problem, unspecified, not applicable"}
            ]


df = pd.read_csv(r"C:\users\steve_000\Documents\SAVSNET,sample,vet,data_12Sept2023.csv", encoding='unicode_escape', nrows=28)
results = []
truth = df[['Diagnosis', 'Certainty', 'Severity', 'Concurrency']].apply(lambda x: x.astype(str).str.upper())

for index, row in df.iterrows():
    messages = prompt.copy()
    messages.append({"role": "user", "content": row['Narrative']})
    results.append(query_chatgpt(messages))


resultDf = pd.DataFrame.from_dict(results)[['Diagnosis', 'Certainty', 'Severity', 'Concurrency']].apply(lambda x: x.astype(str).str.upper())

diffs = np.where(resultDf['Diagnosis'] != truth['Diagnosis'])
print('Diagnosis Accuracy', (len(results) - diffs[0].size) / len(results))
diffs = np.where(resultDf['Certainty'] != truth['Certainty'])
print('Certainty Accuracy', (len(results) - diffs[0].size) / len(results))
diffs = np.where(resultDf['Severity'] != truth['Severity'])
print('Severity Accuracy', (len(results) - diffs[0].size) / len(results))
diffs = np.where(resultDf['Concurrency'] != truth['Concurrency'])
print('Concurrency Accuracy', (len(results) - diffs[0].size) / len(results))