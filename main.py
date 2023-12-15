
import panel as pn
import pandas as pd
import bokeh.plotting as bp
from bokeh.palettes import Spectral4

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import geopandas as gpd
import hvplot.pandas
from datetime import datetime
import folium
from panel.interact import interact
pn.extension()
pn.extension('tabulator')


from bokeh.transform import cumsum
from bokeh.plotting import figure
import plotly.express as px

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable
import time


# cache data to improve dashboard performance
if 'data' not in pn.state.cache.keys():
    df = pd.read_excel('data/processed/dashboard.xlsx')
    pn.state.cache['data'] = df.copy()
else: 
    df = pn.state.cache['data']


# Make DataFrame Pipeline Interactive
idf = df.interactive()
df['year'] = df['Fecha Creacion'].dt.year
df['Ciudad'].unique()

df.info()


# Define widgets
year_slider = pn.widgets.IntSlider(name='Year', start=2011, end=2021, value=2021, width=800)
yaxis_year_world = pn.widgets.RadioButtonGroup(name='Y axis', options=['Monto', 'Precio'], button_type='success')

# Esta función actualiza el gráfico en función del año seleccionado y el eje Y
@pn.depends(year_slider.param.value, yaxis_year_world.param.value)
def update_plot(year, yaxis):
    # Filtrar los datos según el año seleccionado
    df_year = df[df['year'] <= year]
    # Calcular las ventas totales y el precio promedio para cada año
    sales_quantity = df_year.groupby('year')['Monto'].sum() / 1000
    sales_price = df_year.groupby('year')['Precio'].mean()
    # Combinar los datos de ventas totales y precio promedio para cada año
    sales_data = pd.concat([sales_quantity, sales_price], axis=1)
    sales_data = sales_data.reset_index()
    sales_data = sales_data.melt(id_vars='year', var_name='yaxis', value_name='value')
    # Filtrar los datos de ventas según el eje Y seleccionado
    sales_data = sales_data[sales_data['yaxis'] == yaxis]
    # Crear el gráfico
    p = bp.figure(title='Facturación por año' if yaxis == 'Monto' else 'Precio por año',
                  width=800, height=400, x_range=(2011, 2021),
                  y_range=(0, sales_quantity.max() * 1.1) if yaxis == 'Monto' else (0, sales_price.max() * 1.1))
    p.xaxis.axis_label = "Year"
    p.yaxis.axis_label = "Facturación (en Miles)" if yaxis == 'Monto' else "Price"
    p.line(x='year', y='value', source=sales_data, color=Spectral4[0], line_width=2)
    return pn.pane.Bokeh(p)



# Define widgets
year_slider = pn.widgets.IntSlider(name='Year', start=2011, end=2021, value=2021, width=800)
yaxis_year_world = pn.widgets.RadioButtonGroup(name='Y axis', options=['Monto', 'Precio'], button_type='success')

@pn.depends(year_slider.param.value, yaxis_year_world.param.value)
def update_plot(year, yaxis):
    # Filter the data based on the selected year
    df_year = df[df['year'] <= year]
    # Calculate total sales and average price for each year
    sales_quantity = df_year.groupby('year')['Monto'].sum() / 1000
    sales_price = df_year.groupby('year')['Precio'].mean()
    # Combine the total sales and average price data for each year
    sales_data = pd.concat([sales_quantity, sales_price], axis=1)
    sales_data = sales_data.reset_index()
    sales_data = sales_data.melt(id_vars='year', var_name='yaxis', value_name='value')
    # Filter the sales data based on the selected y-axis
    sales_data = sales_data[sales_data['yaxis'] == yaxis]
    # Create the plot
    if yaxis == 'Monto':
        p = bp.figure(title='Facturación por año', width=800, height=400, x_range=(2011, 2021), y_range=(0, sales_quantity.max()*1.1))

        p.yaxis.axis_label = "Facturación (en Miles)"
        p.line(x='year', y='value', source=sales_data, color=Spectral4[0], line_width=2)
    else:
        p = bp.figure(title='Price by Year', width=800, height=400, x_range=(2011, 2021), y_range=(0, sales_price.max()*1.1))

        p.yaxis.axis_label = "Price"
        p.xaxis.axis_label = "Year"
        p.line(x='year', y='value', source=sales_data, color=Spectral4[0], line_width=2)
    return pn.pane.Bokeh(p)
# Define widgets
year_slider_1 = pn.widgets.IntSlider(name='Year', start=2011, end=2021, value=2021, width=800)
#yaxis_radio_button_1 = pn.widgets.RadioButtonGroup(name='Y axis', options=['Monto'], button_type='success')

@pn.depends(year_slider_1.param.value)
def show_top_products_table(year):
    # Filter the data based on the selected year
    df_year = df[df['year'] == year]
    # Calculate total sales by product for the year
    sales_data = df_year.groupby(['Producto'])['Monto'].sum().reset_index(name='Monto')
    # Get the top 10 selling products
    top_products = sales_data.nlargest(10, 'Monto')
    # Display the top 10 selling products in a table with title
    return pn.Column(pn.pane.Markdown("# Top Productos"), pn.pane.DataFrame(top_products, width=800))

@pn.depends(year_slider_1.param.value)
def show_top_products_pie_chart(year):
    # Filter the data based on the selected year
    df_year = df[df['year'] == year]
    # Calculate total sales by product for the year
    sales_data = df_year.groupby(['Producto'])['Monto'].sum().reset_index(name='Monto')
    # Get the top 10 selling products
    top_products = sales_data.nlargest(10, 'Monto')
    # Create a pie chart of the top 10 selling products
    fig = px.pie(top_products, values='Monto', names='Producto', title='Top 10 Selling Products',template="plotly_dark")
    return pn.pane.Plotly(fig)

def show_top_risk_table():
    # Filter the data based on the 'risk' column with value 0
    df_risk = df[df['risk'] == 0]
    # Calculate total sales by 'objective_days' column
    risk_data = df_risk.groupby(['objective_days','Debt_rating'])['Monto'].sum().reset_index(name='Monto')
    # Sort the table by 'objective_days'
    risk_data = risk_data.sort_values('objective_days')
    # Display all the data
    return pn.Column(pn.pane.Markdown("# Condición de créditos por días de atraso"), pn.pane.DataFrame(risk_data, width=800))

@pn.depends(year_slider_1.param.value)
def show_top_Vendedor_table(year):
    # Filter the data based on the selected year
    df_year = df[df['year'] == year]
    # Calculate total sales by product for the year
    sales_data = df_year.groupby(['Vendedor'])['Monto'].sum().reset_index(name='Monto')
    # Get the top 10 selling products
    top_vendedor = sales_data.nlargest(3, 'Monto')
    # Display the top 10 selling products in a table with title
    return pn.Column(pn.pane.Markdown("# Top Vendedor"), pn.pane.DataFrame(top_vendedor, width=800))

@pn.depends(year_slider_1.param.value)
def show_top_Vendedor_bar_chart(year):
    # Filter the data based on the selected year
    df_year = df[df['year'] == year]
    # Calculate total sales by vendor for the year
    sales_data = df_year.groupby(['Vendedor'])['Monto'].sum().reset_index(name='Monto')
    # Get the top 10 selling vendors
    top_vendors = sales_data.nlargest(10, 'Monto')
    # Create a bar chart of the top 10 selling vendors
    fig = px.bar(top_vendors, x='Vendedor', y='Monto', title=f'Top 10 Vendors for {year}', 
                 labels={'Vendedor': 'Vendor', 'Monto': 'Total Sales'},template="plotly_dark")
    fig.update_layout(xaxis={'title': 'Year'})
    return pn.pane.Plotly(fig)

def show_acctions_table():
    # Filter the data based on the 'risk' column with value 0
    df_risk = df[df['risk'] == 0]
    # Calculate total sales by 'objective_days' column
    acctions_data = df_risk.groupby(['TransaccionID','Acctions_rating'])['Monto'].sum().reset_index(name='Monto')
    # Sort the table by 'objective_days'
    acctions_data = acctions_data.sort_values('Acctions_rating')
    # Display all the data
    return pn.Column(pn.pane.Markdown("#Acciones para créditos morosos"), pn.pane.DataFrame(acctions_data, width=800))

def show_ratio_prc():
    # Filter the data based on the 'risk' column
    df_risk_0 = df[df['risk'] == 0]
    df_risk_1 = df[df['risk'] == 1]

    # Calculate total sales and porcobrar by 'risk' column
    porcobrar = round(df_risk_0['Monto'].sum()/100, 1)
    ventas = round(df_risk_1['Monto'].sum() / 1.18/100, 1)

    # Calculate rotacion_por_cobrar and prc
    rotacion_por_cobrar = round(ventas / porcobrar, 1)
    prc = round(360 / rotacion_por_cobrar, 1)

    # Display all the data
    data = {'Porcobrar': [porcobrar], 'Ventas': [ventas], 'Rotación por Cobrar': [rotacion_por_cobrar], 'PRC': [prc]}
    actions_data = pd.DataFrame(data, columns=['Porcobrar', 'Ventas', 'Rotación por Cobrar', 'PRC'])
    return pn.Column(pn.pane.Markdown("#Ratio de Periodo de Recuperación por cobrar"), pn.pane.DataFrame(actions_data, width=800))

def show_ciudad_table():
    # Replace NaN values in 'Ciudad' column with 'Desconocido'
    df['Ciudad'] = df['Ciudad'].fillna('Desconocido')

    # Calculate total sales by 'Ciudad' and 'voucher_condition' columns
    data_ciudad = df.groupby(['Ciudad', 'voucher_condition'])['Monto'].sum().reset_index(name='Monto')

    # Sort the table by 'Ciudad'
    data_ciudad = data_ciudad.sort_values('Ciudad')

    # Display all the data
    return pn.Column(pn.pane.Markdown("#Ciudades y facturacion"), pn.pane.DataFrame(data_ciudad, width=800))
ACCENT = "#BB2649"
RED = "#D94467"
GREEN = "#5AD534"
# Define tus funciones y variables aquí...
template = pn.template.FastListTemplate(
    title='Enterprise dashboard 2011-2021', 
    sidebar=[pn.pane.Markdown("# Enterprise dashboard 2011-2021"), 
             pn.pane.Markdown("#### This compiled dataset pulled from four other datasets linked by time and place, and was built to find signals correlated to increased")], 
    accent_base_color=ACCENT,
    header_background=ACCENT,
    #prevent_collision=True,
    #save_layout=True,
    theme_toggle=False,
    theme='dark',
    #row_height=160,
    main=[
        year_slider,  
        yaxis_year_world, 
        pn.Row(pn.Column(update_plot, width=900), 
               pn.Column(show_top_risk_table, width=400)),
        
        year_slider_1,
        #yaxis_radio_button_1,
        pn.Row(pn.Column(show_top_products_table, width=900), 
               pn.Column(show_top_products_pie_chart, width=600)),
        pn.Row(pn.Column(show_top_Vendedor_table, width=900), 
               pn.Column(show_top_Vendedor_bar_chart, width=600)),
        
        pn.Row(pn.Column(show_acctions_table, width=900), 
               pn.Column(show_ratio_prc, width=600)),
        
        pn.Row(pn.Column(show_ciudad_table, width=900))
 
    ]
)

template.show()



