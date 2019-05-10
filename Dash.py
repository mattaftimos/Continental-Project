import base64
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import plotly.tools as tls
import datetime
import os
from flask_caching import Cache
import dash_core_components as dcc
import plotly.graph_objs as go
import plotly
import pandas as pd
import numpy as np
import re
import dash_auth
import os
from os import listdir
from os.path import isfile, join
from plotly import tools
import configparser

#########LOGIN READING###########
valid_Combinations = pd.read_csv(
    'LoginInformation/Login.csv')  # reading information from the csv file created within the folder named "LoginInformation"
valid_User_Combinations = valid_Combinations.values.tolist()  # assigning information form the csv file into a list so we can display it later on

##############LOGIN AUTH#####################
VALID_USERNAME_PASSWORD_PAIRS = valid_User_Combinations  # VALID_USERNAME_PASSWORD_PAIRS is a variable from Dash that holds the necessary information for username's and passpwords in order to open the website

image_filename = 'Insight.png'  # reading Insight Logo
encoded_image = base64.b64encode(open(image_filename, 'rb').read())  # encoding image so we can call it later
cwd = os.getcwd()  # working directory of the process

path = os.path.join(cwd, "cleaneddata")  # assigning path to the concatentaton of cleandata and cwd

datasetfiles = [f for f in listdir(path) if isfile(join(path, f)) if f.endswith(".dat")]  # joining

datasetnameslist = [f.split('.dat', 1)[0] for f in datasetfiles]  # splitting

dictlistdatasets = [{'label': f, 'value': f} for f in datasetnameslist]

# Global Variables for Settings
selectedColor = 'Selected Color'  # used a a string to display a header

app = dash.Dash('auth')
############## AUTH CODE####################
auth = dash_auth.BasicAuth(
    app,
    VALID_USERNAME_PASSWORD_PAIRS  # inserting the information from the csv within the the "LoginInformation" folder
)

############## INITIALIZATION OF CACHE CODE#####################
cache = Cache(app.server, config={
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': 'cache'
})
########### END OF INITIALIZATION OF CACHE CODE####################


Changedtable = False
server = app.server
app.scripts.config.serve_locally = True
timeout = 3600  # timeout for cache
optionsDataset = dictlistdatasets


def compute_expensive_data():
    return str(datetime.datetime.now())


# get dataframe based on string
@cache.memoize(timeout=timeout)  # cache
def fetch_dataframe(df_name):
    return pd.read_pickle('cleaneddata/' + df_name + '.dat')


# getting dataframe name for timeseries in tab 2
def fetch_timeseries_dataframe(df_name):
    df_name = df_name.replace('/', '-')
    return pd.read_pickle('cleaneddata/Timeseries/' + df_name + '.dat')


@cache.memoize(timeout=timeout)  # cache
def lengthofdataset(data):
    rows, columns = data.shape
    return rows


# code to see how much percentage of a dataset you wish to laod
def amountofrowsfrompercentage(number, data):
    return data[:int(lengthofdataset(data) * (number / 100))]


# Dataset Columns
@cache.memoize(timeout=timeout)
def load_dataset_columns(dataset):
    optionsColumns = fetch_dataframe(dataset).columns.values
    return [
        {'label': column, 'value': column}
        for column in optionsColumns
    ]


@cache.memoize(timeout=timeout)
def load_cost_centers():
    optionsAreas = (fetch_dataframe('mabsences')['Cost Center']).unique().tolist()
    return [
        {'label': area, 'value': area}
        for area in optionsAreas
    ]


app.layout = html.Div(
    [
        # Hidden div inside the app that stores the intermediate value
        html.Div(id='intermediate-value-pgc', style={'display': 'none'}),  # pgc stands for primary graph color
        html.Div(id='intermediate-value-sgc', style={'display': 'none'}),  # sgc is secondary graph color
        ##################### tabs ####################
        html.Div([
            html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '10vw'}),
            # importing Insight logo
            # Declaring Tabs below
            dcc.Tabs(
                tabs=[
                    {'label': 'Visualize Data', 'value': 1},  # tab1
                    {'label': 'Predictive Model', 'value': 2},  # tab2
                    {'label': 'Settings', 'value': 3},  # tab3
                ],
                value=1,  # default value to tab 1
                id='tabs',
                vertical=True,  # making the tabs vertical tabs
                style={
                    'height': '100vh',
                    'borderRight': 'thin lightgrey solid',
                    'textAlign': 'left'
                }

            )
        ], style={'width': '10%', 'float': 'left'}),

        html.Div([
            ################################ tab1 ################################
            html.Div([
                html.H6('Dataset'),  # title
                ##Dropdown for dataset type ( from the possible datasets put inside the folder[cleaneddate])
                dcc.Dropdown(
                    id='datasets-dropdown',
                    options=optionsDataset,
                    value=datasetnameslist[0],
                ),

                html.H6('Load percent of dataset'),
                # incase files are large, the user doesnt have to wait for millions of fields to load unless they need too
                dcc.Input(
                    placeholder='Enter a percentage...',
                    # asks the user to put a percentage of information from a file they selected,
                    id='percentage-datatable',
                    type='number',
                    value=75,
                    max=100,
                    min=0
                ),

                # code to show or hide the filterable data
                html.Details([
                    html.Summary('DataTable'),
                    html.Div(dt.DataTable(
                        rows=amountofrowsfrompercentage(75, fetch_dataframe(datasetnameslist[0])).to_dict('records'),
                        columns=sorted(fetch_dataframe(datasetnameslist[0]).columns),
                        row_selectable=True,
                        filterable=True,
                        sortable=True,
                        selected_row_indices=[],
                        id='datatable'
                    ))
                ]),
                html.Div(id='selected-indexes'),

                #################### START of first graph ####################
                dcc.Dropdown(id='columns-dropdown', multi=True),
                html.Div(id='display-selected-values'),

                ############# User choosing Graph Type ##################
                dcc.RadioItems(
                    id='choose-graph-type',
                    options=[{'label': i, 'value': i} for i in ['Bar', 'Scatter', 'Pie']],
                    value='Bar',
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Graph(id='indicator-graphic', style={'width': '87vw'}),
                #################### END of first graph ####################

                html.Hr()

                # #################### START of second graph ###################
                # dcc.Dropdown(
                #     id='datasets-dropdown2',
                #     options=optionsDataset,
                #     value=datasetnameslist[0],
                # ),
                # html.Div([
                #     html.Div([
                #         html.Div('X-Axis', style={'padding-left': '10vw'}),
                #         dcc.Dropdown(id='x-axis-column')
                #     ], style={'width': '49%', 'display': 'inline-block'}),
                #
                #     html.Div([
                #         html.Div('Y-Axis', style={'padding-left': '10vw'}),
                #         dcc.Dropdown(id='y-axis-column', )
                #     ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
                # ], style={
                #     'borderBottom': 'thin lightgrey solid',
                #     'backgroundColor': 'rgb(250, 250, 250)',
                #     'padding': '1vw 1vw'
                # }),
                # dcc.Graph(id='indicator-graphic2', style={'width': '87vw'})
                # #################### END of second graph ###################

            ], id='tab1div', style={'display': 'none'}),  # end of div for tab1

            ################################ END for tab1 ################################

            ################################ START for tab2 ################################
            html.Div([

                html.H4('Absences Predictive Model'),
                # html.H6('Prediction Percentage'),
                # # allowing the user to input a percentage which will impact the predictive model
                # dcc.Input(
                #     placeholder='Enter a percentage...',
                #     id='prediction-over-under',
                #     type='number',
                #     value=0
                # ),
                html.H6('Select Cost Center'),
                # choosing which cost centers they want to include within the predictive model
                dcc.Dropdown(
                    id='prediction-areas',
                    options=load_cost_centers(),
                    multi=True
                ),
                dcc.RadioItems(  # giving the user options to choose between types of graphs for their liking
                    id='graph2-options',
                    options=[{'label': i, 'value': i} for i in ['Mesh-Grid', 'Stacked']],
                    value='Mesh-Grid',
                    labelStyle={'display': 'inline-block'}
                ),
                html.H6('Smoothness'),
                # function to allow smoothing for the graphs, allowing the user to adjust how sharp the data will look on their graph

                dcc.Slider(  # slider for smoothness
                    id='slider',
                    min=-0,
                    max=1.3,
                    step=0.1,
                    value=0,
                ),
                html.Div(id='slider-output-container'),

                dcc.Graph(id='timeseries'),
                html.Div(id='display-selected-values-prediction'),

            ], id='tab2div', style={'display': 'none', 'padding-left': '1vw', 'padding-right': '1vw'}),
            ################################ END for tab2 ################################

            ################################ START for tab3 ################################
            html.Div([

                html.H4('Graph Settings'),  # title

                "Primary Graph Color: ",
                dcc.Input(
                    placeholder='Enter a Hex color...',
                    # allows the user to input a hex color of their liking which will change the color of the graphin within tab1 and tab2
                    id='graph-color',
                    type='text',
                    value=''
                ),

                # Graph Color Preview Output
                html.H5(selectedColor, id='graph-color-preview'),
                html.Br(),

                # Callback Output for graph color
                html.Div(id='graph-color-output'),

                "Secondary Graph Color: ", dcc.Input(
                    placeholder='Enter a Hex color...',
                    # this function will change the color of the data the user wants to filter from the data they select/deselect from the table in tab1
                    id='secondary-graph-color',
                    type='text',
                    value=''
                ),

                # Secondary Graph Color Preview Output
                html.H5(selectedColor, id='secondary-graph-color-preview'),
                html.Br(),

                # Callback Output for secondary graph color
                html.Div(id='secondary-graph-color-output'),
                html.H6('Clear Cache'),
                html.Button('Clear Cache', id='cache-button', n_clicks=0),
                # button to allow the user to clear cache if they dont want

                # Callback Output for Cache Button
                html.Div(id='cache-button-output'),

                html.H6('Config File'),
                # button allowing the user to save their changes like graph color to a confi which will be loaded when the website is reloaded

                # Config Button
                html.Button('Create Config file', id='config-button', n_clicks=0),

                # Callback Output for Config Button
                html.Div(id='config-button-output')

                ################################ END for tab3 ################################

            ], id='tab3div', style={'display': 'none'})
        ], style={'width': '89%', 'float': 'right'})

    ]  # end of main HTML.DIV
    , style={'width': '99.2%', 'display': 'inline-block'})


############## end of layout ##################


################ START for tab callbacks ################
@app.callback(Output('tab1div', 'style'), [Input('tabs', 'value')])  # callback for tab1
def display_tab1(tab):
    return {'display': 'block' if tab == 1 else 'none'}


@app.callback(Output('tab2div', 'style'), [Input('tabs', 'value')])  # callback for tab2
def display_tab2(tab):
    return {'display': 'block' if tab == 2 else 'none'}


@app.callback(Output('tab3div', 'style'), [Input('tabs', 'value')])  # callback for tab3
def display_tab3(tab):
    return {'display': 'block' if tab == 3 else 'none'}


################ END for tab callbacks ################


################ START for datatable( user can select certain information and filter them out) ################
@app.callback(
    dash.dependencies.Output('datatable', 'selected_row_indices'),
    [Input('indicator-graphic', 'clickData')],
    [State('datatable', 'selected_row_indices')])
@cache.memoize(timeout=timeout)
# update for selected rows
def update_selected_row_indices(clickData, selected_row_indices):
    if Changedtable:
        return []
    if clickData:
        for point in clickData['points']:
            if point['pointNumber'] in selected_row_indices:
                selected_row_indices.remove(point['pointNumber'])
            else:
                selected_row_indices.append(point['pointNumber'])
    return selected_row_indices


@app.callback(
    Output('datatable', 'rows'),
    [dash.dependencies.Input('datasets-dropdown', 'value'),
     Input('percentage-datatable', 'value')])
# update for the percentage to load the dataset
def update_table(dataset, percentage):
    global Changedtable
    Changedtable = True
    return amountofrowsfrompercentage(percentage, fetch_dataframe(dataset)).to_dict('records')


@app.callback(
    Output('percentage-datatable', 'value'),
    [dash.dependencies.Input('datasets-dropdown', 'value')])
@cache.memoize(timeout=timeout)
def update_defaultpercentagevalue(dataset):
    length = lengthofdataset(fetch_dataframe(dataset))
    if length* .90 > 15000:
        return int((15000 / length) * 100)
    return 100


@app.callback(
    Output('columns-dropdown', 'value'),
    [Input('datasets-dropdown', 'value')])
def clearoptions(data):
    return []


@app.callback(
    Output('datatable', 'columns'),
    [dash.dependencies.Input('datasets-dropdown', 'value')])
@cache.memoize(timeout=timeout)
def update_table(dataset):
    return fetch_dataframe(dataset).columns


################ END for datatable( user can select certain information and filter them out) ################


############# START for dropdown callbacks ##############
@app.callback(
    dash.dependencies.Output('columns-dropdown', 'options'),
    [dash.dependencies.Input('datasets-dropdown', 'value')])
@cache.memoize(timeout=timeout)
def set_columns_options(selected_dataset):
    return load_dataset_columns(selected_dataset)


# @app.callback(
#     dash.dependencies.Output('columns-dropdown', 'value'),
#     [dash.dependencies.Input('columns-dropdown', 'options')])
@cache.memoize(timeout=timeout)
# def set_columns_value(available_options):
#     return available_options[0]['value']

@app.callback(
    dash.dependencies.Output('display-selected-values', 'children'),
    [dash.dependencies.Input('datasets-dropdown', 'value'),
     dash.dependencies.Input('columns-dropdown', 'value')])
@cache.memoize(timeout=timeout)
def set_display_children(selected_dataset, selected_column):
    message = ""
    if isinstance(selected_column, list):
        numSelected = len(selected_column)
        if numSelected > 1:
            message += "Selected " + str(numSelected) + " items:"
            for item in selected_column:
                message += ' ' + item + ','
            message = message[:-1] + ' in ' + str(selected_dataset)
        elif numSelected == 1:
            message = "Selected " + str(selected_column[0]) + ' in ' + str(selected_dataset)
    else:
        message = "Selected " + str(selected_column) + ' in ' + str(selected_dataset)
    return u'{}'.format(message)


############# END for dropdown callbacks ##############

@cache.memoize(timeout=timeout)
def dataframe_from_rows(rows):
    return pd.DataFrame(rows)


def getUniqueCounts(selected_column, data):
    return np.unique(data[selected_column].astype(str),
                     return_counts=True)  # gets unique amount of data from the selected columns and stores it as a list


## Function to set the color of graphs, pgc = primary graph color, sgc = secondary graph color
def setcolormarker(data, selected_row_indices, pgc, sgc):
    if type(pgc) is None:
        pgc = '#7FDBFF'
    if type(sgc) is None:
        sgc = '#FA8072'
    marker = {'color': [pgc] * len(data)}
    for i in (selected_row_indices or []):
        marker['color'][i] = sgc
    return marker


@cache.memoize(timeout=timeout)
def tolist(data):
    return data.tolist()


############### START for graph callback ################
@app.callback(
    Output('indicator-graphic', 'figure'),  # indicator-graphic is the first graph
    [Input('datasets-dropdown', 'value'),
     Input('datatable', 'rows'),
     Input('columns-dropdown', 'value'),
     Input('choose-graph-type', 'value'),
     Input('datatable', 'selected_row_indices'),
     Input('intermediate-value-pgc', 'children'),
     Input('intermediate-value-sgc', 'children')])
def update_graph(datao, rows, selected_column, graph_type, selected_row_indices, pgc, sgc):
    if isinstance(selected_column, list):
        numSelected = len(selected_column)
    else:
        numSelected = 1
    if numSelected == 0:
        N = 10
        random_x = np.linspace(0, 1, N)
        random_y0 = np.random.randn(N) + 5
        random_y1 = np.random.randn(N)
        random_y2 = np.random.randn(N) - 5

        # Create traces
        trace0 = go.Scatter(
            x=random_x,
            y=random_y0,
            mode='lines',
            name='lines'
        )
        trace1 = go.Scatter(
            x=random_x,
            y=random_y1,
            mode='lines+markers',
            name='lines+markers'
        )
        trace2 = go.Scatter(
            x=random_x,
            y=random_y2,
            mode='markers',
            name='markers'
        )
        data = [trace0, trace1, trace2]
        layout = go.Layout(title="Please select a column", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})
        return go.Figure(data=data, layout=layout)
    dff = dataframe_from_rows(rows)

    if numSelected == 1:
        index, counts = getUniqueCounts(selected_column[0], dff)

        if graph_type == 'Bar':
            marker = setcolormarker(dff, selected_row_indices, pgc, sgc)
            return {
                'data': [
                    go.Bar(
                        x=tolist(index),
                        y=tolist(counts),
                        name=selected_column,  # title of the graph is the selected column
                        marker=marker  # returns the user color
                    )
                ],
                'layout': go.Layout(
                    xaxis={
                        'title': selected_column,
                    }
                )
            }
        if graph_type == 'Pie':
            return {
                'data': [
                    go.Pie(
                        labels=tolist(index),
                        values=tolist(counts),
                        name=selected_column,
                    )
                ],

            }
        if graph_type == 'Scatter':
            marker = setcolormarker(dff, selected_row_indices, pgc, sgc)
            return {
                'data': [
                    go.Scattergl(
                        x=tolist(index),
                        y=tolist(counts),
                        name=selected_column,
                        marker=marker
                    )
                ],
                'layout': go.Layout(
                    xaxis={
                        'title': selected_column,
                    }
                )
            }
    else:  # multiple columns selected
        if numSelected == 2:
            marker = setcolormarker(dff, selected_row_indices, pgc, sgc)
            return {
                'data': [
                    go.Scattergl(
                        x=dff[selected_column[0]],
                        y=dff[selected_column[1]],
                        name=selected_column[0] + " to " + selected_column[1],
                        marker=marker,
                        mode='markers'
                    )
                ],
                'layout': go.Layout(
                    xaxis={
                        'title': selected_column[0],
                    },
                    yaxis={
                        'title': selected_column[1],
                    }
                )
            }
        elif numSelected == 3:
            marker = setcolormarker(dff, selected_row_indices, pgc, sgc)
            x = dff[selected_column[0]]
            y = dff[selected_column[1]]
            z = dff[selected_column[2]]
            marker['opacity'] = 0.8
            marker['size'] = 12
            return {
                'data': [
                    go.Scatter3d(
                        x=x,
                        y=y,
                        z=z,
                        name=selected_column[0] + " to " + selected_column[1] + " and " + selected_column[2],
                        marker=marker,
                        mode='markers'
                    )
                ],
                'layout': go.Layout(
                    margin=dict(
                        l=0,
                        r=0,
                        b=0,
                        t=0
                    )

                )
            }


############### END for graph callback ################

################ START for Timessereies(tab2) ######################
@app.callback(
    Output('timeseries', 'figure'),
    [Input('prediction-areas', 'value'),
     Input('graph2-options', 'value'),
     Input('slider', 'value')])
def update_graph(costcenter, graph_value, smoothness):
    if not costcenter:
        N = 10
        random_x = np.linspace(0, 1, N)
        random_y0 = np.random.randn(N) + 5
        random_y1 = np.random.randn(N)
        random_y2 = np.random.randn(N) - 5

        # Create traces
        trace0 = go.Scatter(
            x=random_x,
            y=random_y0,
            mode='lines',
            name='lines'
        )
        trace1 = go.Scatter(
            x=random_x,
            y=random_y1,
            mode='lines+markers',
            name='lines+markers'
        )
        trace2 = go.Scatter(
            x=random_x,
            y=random_y2,
            mode='markers',
            name='markers'
        )
        data = [trace0, trace1, trace2]
        layout = go.Layout(title="Please select a column", xaxis={'title': 'x1'}, yaxis={'title': 'x2'})
        return go.Figure(data=data, layout=layout)
    if graph_value == 'Mesh-Grid':
        fig = tools.make_subplots(rows=3, cols=1, specs=[[{}], [{}], [{}]],
                                  shared_xaxes=True, shared_yaxes=True,
                                  vertical_spacing=0.001, print_grid=False)
        for x in costcenter:
            df = fetch_timeseries_dataframe(x)
            trace = go.Scatter(
                x=df.index,
                y=df['absentees'],
                name=x,
                opacity=0.8,
                line=dict(
                    shape='spline',
                    smoothing=smoothness
                )
            )
            # buttons that allows the user to choose the duration of months for the prediction model
            layout = dict(
                title='Time Series',
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 label='1m',
                                 step='month',
                                 stepmode='backward'),
                            dict(count=3,
                                 label='3m',
                                 step='month',
                                 stepmode='backward'),
                            dict(count=6,
                                 label='6m',
                                 step='month',
                                 stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(),
                    type='date'
                )
            )
            fig.append_trace(trace, 1, 1)
        fig['layout'] = layout
        fig['layout'].update(height=850)
    if graph_value == "Stacked":
        if type(costcenter) == list:
            fig = tools.make_subplots(rows=len(costcenter), cols=1, vertical_spacing=.5 / len(costcenter),print_grid=False)
        if type(costcenter) == str:
            fig = tools.make_subplots(rows=1, cols=1,print_grid=False)
        for x in costcenter:
            df = fetch_timeseries_dataframe(x)
            trace = go.Scatter(
                x=df.index,
                y=df['absentees'],
                name=x,
                opacity=0.8,
                line=dict(
                    shape='spline',
                    smoothing=smoothness
                )

            )
            layout = dict(
                title='Time Series',
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 label='1m',
                                 step='month',
                                 stepmode='backward'),
                            dict(count=6,
                                 label='6m',
                                 step='month',
                                 stepmode='backward'),
                            dict(step='all')
                        ])
                    ),
                    rangeslider=dict(),
                    type='date'
                )
            )
            if type(costcenter) == list:
                fig.append_trace(trace, costcenter.index(x) + 1, 1)
            if type(costcenter) == str:
                fig.append_trace(trace, 1, 1)
        if type(costcenter) == list:
            fig['layout'].update(height=300 * len(costcenter))
    return fig


################ END for Timessereies(tab2) ######################

############# START for Prediction Callback #############


# Slider for predictive model smoothness
@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('slider', 'value')])
def update_output(value):
    return 'Smoothness Level = {}'.format(int(value * 10))


############# END for Prediction Callback #############

############## START for Settings Callbacks###############
# Graph Color
@app.callback(Output('intermediate-value-pgc', 'children'),
              [Input(component_id='graph-color', component_property='value')])
def get_graphColor_value(input_value):
    # Regex to test if string is a Hex Color Code
    isHex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', input_value)

    if isHex and not os.path.isfile('graphColors.ini'):
        return input_value

    elif os.path.isfile('graphColors.ini'):  # If the config file exists
        config = configparser.ConfigParser()  # Start the config parser
        config.read('graphColors.ini')  # Read the config file

        # If input value is a hex and it is different from the value saved in the config
        if isHex and (input_value != config['COLORS']['graphColor']):
            return input_value  # return that new value
        else:
            return config['COLORS']['graphColor']  # Return the saved color from the config

    else:
        # Some Default Hex Value
        return '#7FDBFF'


# Secondary Graph Color
@app.callback(Output('intermediate-value-sgc', 'children'),
              [Input(component_id='secondary-graph-color', component_property='value')])
def get_secondary_graphColor_value(input_value):
    # Regex to test if string is a Hex Color Code
    isHex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', input_value)

    if isHex and not os.path.isfile('graphColors.ini'):
        return input_value

    elif os.path.isfile('graphColors.ini'):  # If the config file exists
        config = configparser.ConfigParser()  # Start the config parser
        config.read('graphColors.ini')  # Read the config file

        # If input value is a hex and it is different from the value saved in the config
        if isHex and (input_value != config['COLORS']['secondaryGraphColor']):
            return input_value  # return that new value
        else:
            return config['COLORS']['secondaryGraphColor']  # Return the saved color from the config

    else:
        # Some Default Hex Value
        return '#FA8072'


# Clear Cache
@app.callback(Output(component_id='cache-button-output', component_property='children'),
              [Input(component_id='cache-button', component_property='n_clicks')])
def clear_cache(num_clicks):
    # Clear Cache
    cache.clear()
    if num_clicks < 1:
        return ''
    else:
        return 'The cache was cleared successfully {} time(s)'.format(num_clicks)


# Graph Color Preview
@app.callback(Output(component_id='graph-color-preview', component_property='style'),
              [Input(component_id='graph-color', component_property='value'),
               Input('intermediate-value-pgc', 'children')])
def update_graphColor_preview(input_value, pgc):
    # Regex to test if string is a Hex Color Code
    isHex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', input_value)

    if isHex:
        return {'background-color': input_value,
                'color': input_value, 'display': 'inline-block', 'margin-left': '2.85vw',
                'width': '5vw', 'height': '5vw', 'font-size': '1.25vw', 'text-align': 'center'}
    else:

        return {'background-color': pgc,
                'color': pgc, 'display': 'inline-block', 'margin-left': '2.85vw',
                'width': '5vw', 'height': '5vw', 'font-size': '1.25vw', 'text-align': 'center'}


# Secondary Graph Color Preview
@app.callback(Output(component_id='secondary-graph-color-preview', component_property='style'),
              [Input(component_id='secondary-graph-color', component_property='value'),
               Input('intermediate-value-sgc', 'children')])
def update_secondary_graphColor_preview(input_value, sgc):
    # Regex to test if string is a Hex Color Code
    isHex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', input_value)

    if isHex:
        return {'background-color': input_value,
                'color': input_value, 'display': 'inline-block', 'margin-left': '2vw',
                'width': '5vw', 'height': '5vw', 'font-size': '1.25vw', 'text-align': 'center'}
    else:

        return {'background-color': sgc,
                'color': sgc, 'display': 'inline-block', 'margin-left': '2vw',
                'width': '5vw', 'height': '5vw', 'font-size': '1.25vw', 'text-align': 'center'}


# Config File
@app.callback(Output(component_id='config-button-output', component_property='children'),
              [Input(component_id='config-button', component_property='n_clicks'),
               Input(component_id='graph-color', component_property='value'),
               Input(component_id='secondary-graph-color', component_property='value'),
               Input(component_id='intermediate-value-pgc', component_property='children'),
               Input(component_id='intermediate-value-sgc', component_property='children')])
def on_click(num_clicks, graphColorValue, secondaryGraphColorValue, pgc, sgc):
    # Check if values are proper Hexs
    isGraphColorHex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', graphColorValue)
    isSecondaryGraphColorHex = re.search(r'^#(?:[0-9a-fA-F]{3}){1,2}$', secondaryGraphColorValue)

    # Set Primary Graph Color and Secondary Graph Color Values
    if isGraphColorHex:
        configGraphColor = graphColorValue
    else:
        configGraphColor = pgc

    if isSecondaryGraphColorHex:
        configSecondaryGraphColor = secondaryGraphColorValue
    else:
        configSecondaryGraphColor = sgc

    if num_clicks > 0:
        # Create config file
        config = configparser.ConfigParser()
        config['COLORS'] = {'graphColor': configGraphColor,
                            'secondaryGraphColor': configSecondaryGraphColor}
        with open('graphColors.ini', 'w') as configfile:
            config.write(configfile)

        return 'Config file was created successfully'


############## END for Settings Callbacks###############


app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)
