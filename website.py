from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)

df = pd.read_pickle('data.pkl')
@app.route('/')
def index():
    # Pass an empty form initially
    return render_template('index.html', form_data={}, results=None)

@app.route('/search', methods=['POST'])
def search():
    # Get form data
    column_name = request.form['column_name']
    min_value = float(request.form['min_value'])
    max_value = float(request.form['max_value'])

    # Filter DataFrame based on user input
    filtered_df = df[df[column_name] >= min_value]
    filtered_df = filtered_df[filtered_df[column_name] <= max_value]

    # Prepare results for the template
    results = filtered_df.to_html(index=False)  # Convert to HTML for display

    return render_template('index.html', form_data=request.form, results=results)

if __name__ == '__main__':
    app.run(debug=True)
