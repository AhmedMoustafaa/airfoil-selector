import pandas as pd
import json
df = pd.read_csv('res.csv')
def parse_json(json_string):
  """
  This function parses a JSON string back into a Python list.

  Args:
      json_string (str): A JSON string representing a list.

  Returns:
      list: The parsed list from the JSON string, or None if parsing fails.
  """
  try:
    return json.loads(json_string)
  except json.JSONDecodeError:
    return None  # Handle potential parsing errors

df['cl'] = df['cl'].apply(parse_json)
df['cd'] = df['cd'].apply(parse_json)

print(df.iloc[1,:])