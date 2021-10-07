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
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.offline import plot
import pandas as pd
import numpy as np

import datetime as dt

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

# colors = {
#     'background': '#111111',
#     'text': '#7FDBFF'
# }

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

df2 = pd.DataFrame({
    "Other": ["Some", "Data"],
    "Table": ["More", "Data"],
    })

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

longdata = np.random.randn(1000000)
longfig = px.line(x=np.arange(longdata.size),y=longdata)
longfig.update_yaxes(fixedrange=True)

table1 = dash_table.DataTable(
    id='table',
    columns=[{"name": i, "id": i} for i in df.columns],
    data=df.to_dict('records'),
)

table2 = dash_table.DataTable(
    id='table2',
    columns=[{"name": i, "id": i} for i in df2.columns],
    data=df2.to_dict('records'),
)

# use tabs for each table?



# use dbc layouts?
row = dbc.Row(
    [
     # Column 1
    dbc.Col(dbc.Card(
        dbc.CardBody(
            dcc.Graph(
                id='example-graph-2',
                figure=fig
            )    
        )
    )),
    # Column 2
    dbc.Col(dbc.Card(
        dbc.CardBody(
            dcc.Graph(
                id='example-graph-3',
                figure=longfig  
            )    
        )    
    ))
    
    
    ]
)

navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Page 1", href="#")),
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="NavbarSimple",
    brand_href="#"
)

app.layout = html.Div(children=[
    navbar,
    
    html.H1(
        children='Hello Dash',
        style={
            'textAlign': 'center',
            # 'color': colors['text']
        }
    ),

    html.Div(children='Dash: A web application framework for Python.', style={
        'textAlign': 'center',
        # 'color': colors['text']
    }),
    
    row,
    
    dbc.Tabs(
        [
            dbc.Tab(table1, label="Tab 1"),
            dbc.Tab(table2, label="Tab 2")
         
        ]
    ),
    
    html.Div(
        id='update-text',
        # style = {'color': colors['text']}
    ),
    
    html.Div(
        id='update-text-slow',
        # style = {'color': colors['text']}
    ),
    
    dcc.Interval(
        id='interval-component',
        interval=500,  # in ms
        n_intervals=0
    ),
    
    dcc.Interval(
        id='slow-interval-component',
        interval=5000, # in ms
        n_intervals=0
    )
    
])

@app.callback(Output('update-text-slow', 'children'),
              Input('slow-interval-component', 'n_intervals'))
def update_text(n):
    return [html.Span('(Slow) Time Now: %.6f\nIntervals: %d' % (dt.datetime.utcnow().timestamp(), n))]

@app.callback(Output('update-text', 'children'),
              Input('interval-component', 'n_intervals'))
def update_text(n):
    return [html.Span('Time Now: %.6f\nIntervals: %d' % (dt.datetime.utcnow().timestamp(), n))]

if __name__ == '__main__':
    app.run_server(debug=True)
