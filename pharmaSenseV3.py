import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import logging
import dash_bootstrap_components as dbc

logging.set_verbosity_error()

# Read CSV file
file_path = './data/sampleSales.csv'
df = pd.read_csv(file_path)

# Global variable for chart specifications
chart_specs = {
    'brand': {'figure': px.bar(df.sort_values(by='Sales', ascending=False), x='Product', y='Sales',
                               title='Sales by Product')},
    'geoCity': {'figure': px.bar(df.sort_values(by='Sales', ascending=False).nlargest(25, 'Sales'), x='City', y='Sales',
                                 title='Sales by City')},
    'geoState': {
        'figure': px.bar(df.sort_values(by='Sales', ascending=False).nlargest(25, 'Sales'), x='State', y='Sales',
                         title='Sales by State')},
    'sales': {
        'figure': px.histogram(df.sort_values(by='Sales', ascending=False), x='Sales', title='Sales Distribution')},
    'map': {'figure': px.choropleth(
        df[df['Country'] == 'USA'],
        locations='State',
        locationmode='USA-states',
        color='Sales',
        scope='usa',
        title='USA Map - Sales by State',
        color_continuous_scale=[
            [0, 'red'],  # Red for low sales
            [0.5, 'orange'],  # Amber for medium sales
            [1, 'green']  # Green for high sales
        ]
    )},
    'customer_type': {'figure': px.bar(df.sort_values(by='Sales', ascending=False), x='Customer Type', y='Sales',
                                       title='Sales by Customer Type')},
    'bubble': {'figure': px.scatter(df.nlargest(25, 'Sales'), x='Sales', y='City', size='Sales', color='State',
                                    title='Bubble Chart')},
    'sankey': {'figure': px.parallel_categories(df, dimensions=['Product', 'Customer Type', 'Country'])},
    'sunburst': {'figure': px.sunburst(df, path=['Country', 'State', 'City'], values='Sales', title='Sunburst Chart')},
    'stacked_bar': {'figure': px.bar(
        df,
        x='Product',
        y='Sales',
        color='Customer Type',  # Stack bars based on 'Customer Type'
        title='Stacked Bar Chart by Customer Type'
    )},
    'average_sales_histogram': {'figure': px.histogram(
        df,
        x='Sales',
        title='Histogram of Average Sales by Product',
        color='Product',  # Color by Product for better differentiation
        barmode='overlay',  # Overlay bars for different products
        histfunc='avg'  # Calculate average sales for each product
    )},
    'pie_chart_sales_type': {'figure': px.pie(
        df,
        names='Sales Type',
        title='Pie Chart of Sales Type Distribution',
        hole=0.3  # Set the size of the hole in the middle of the pie chart
    )},

}

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# App layout
app.layout = html.Div(
    [
        dbc.Row(
            dbc.Col(html.H1(className="text-center mb-4", id="pharma-sense-header"), width=12)
        ),

        dbc.Row(
            [
                dbc.Col(
                    dcc.Input(id='input-text', type='text', placeholder='Enter text...', className="mb-2",
                              style={'width': '100%'}),
                    width={'size': 6, 'offset': 3},  # Center the text box with an offset
                ),
                dbc.Col(
                    html.Button('Submit', id='submit-button', n_clicks=0, className="mb-2", style={'width': '100%'}),
                    width={'size': 2, 'offset': 0},  # Move the submit button next to the text box
                ),
            ],
        ),

        dbc.Row(
            dbc.Col(html.Div(id='submit-output'), width=12)
        ),

        dbc.Row(
            dbc.Col(html.Div(id='dynamic-charts',
                             style={'display': 'flex', 'flex-wrap': 'wrap', 'justify-content': 'center'}), width=12)
        ),
    ],
    id='main-container'
)


# Callback for submit button in Section 2
@app.callback(
    [Output('submit-output', 'children'),
     Output('dynamic-charts', 'children'),
     Output('submit-button', 'disabled')],
    Input('submit-button', 'n_clicks'),
    Input('input-text', 'value')
)
def update_output(n_clicks, input_text):
    # Reference to the global chart_specs
    global chart_specs

    if n_clicks is not None and n_clicks > 0:
        # Simulate a delay to show the loading state
        import time
        time.sleep(2)

        # Call the function with user input
        top_chart_ids = get_top_matching_chart_ids(input_text)
        distinct_top_chart_ids = list(set(top_chart_ids))

        # Update chart specs based on distinct_top_chart_ids
        updated_chart_specs = {chart_id: chart_specs[chart_id] for chart_id in distinct_top_chart_ids}

        # Generate charts based on updated_chart_specs
        charts = [dcc.Graph(id=chart_id, **updated_chart_specs[chart_id]) for chart_id in distinct_top_chart_ids]

        # Create a 2x2 layout
        rows = []
        for i in range(0, len(charts), 2):
            row = html.Div(charts[i:i + 2], style={'display': 'flex', 'flex-direction': 'row'})
            rows.append(row)

        # return f'Top matching chart IDs: {", ".join(distinct_top_chart_ids)}', rows, False
        return '', rows, False
    else:
        return '', [], False


def initialize_model():
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    return tokenizer, model


def load_data(file_path):
    return pd.read_csv(file_path)


def get_sentence_embedding(tokenizer, model, sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


def find_top_matching_chart_ids(user_input, data, tokenizer, model, top_k=8):
    chart_ids = data['chartID']
    sentences = data['questions']

    user_embedding = get_sentence_embedding(tokenizer, model, user_input)

    # Calculate cosine similarity between user input and all sentences
    sentence_embeddings = [get_sentence_embedding(tokenizer, model, sent) for sent in sentences]
    similarities = cosine_similarity([user_embedding], sentence_embeddings).flatten()

    # Get indices of top-k most similar sentences
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Return top-k chart IDs
    top_chart_ids = [chart_ids[i] for i in top_indices]
    return top_chart_ids


def get_top_matching_chart_ids(user_input):
    file_path = './data/inputQuestion.csv'
    tokenizer, model = initialize_model()
    data = load_data(file_path)
    top_chart_ids = find_top_matching_chart_ids(user_input, data, tokenizer, model)
    return top_chart_ids


if __name__ == '__main__':
    app.run_server(debug=True)
