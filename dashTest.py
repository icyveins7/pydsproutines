#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 12:41:04 2021

@author: seolubuntu
"""

import dash
from dash import dcc
from dash import html
from dash import dash_table
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import numpy as np

app = dash.Dash(__name__)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

longdata = np.random.randn(1000000)
longfig = px.line(x=np.arange(longdata.size),y=longdata)

fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
)

longfig.update_yaxes(fixedrange=True)

app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        'color': colors['text']
    }),

    dcc.Graph(
        id='example-graph-2',
        figure=fig,
        style={'display':'inline-block',
               'width':500,
               'margin':0}
    ),
    
    dcc.Graph(
        id='example-graph-3',
        figure=longfig,
        style={'display':'inline-block',
               'width':800,
               'margin':0}
    ),
    
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        style_header={'backgroundColor': colors['background']},
        style_cell={
            'backgroundColor': colors['background'],
            'color': colors['text']
        }
    )
    
])

if __name__ == '__main__':
    app.run_server(debug=True)
