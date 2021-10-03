# Imports
# Standard
import numpy as np
import pandas as pd
# Data viz
import matplotlib.pyplot as plt
# Finances
from yahooquery import Ticker
# Other
import streamlit as st


# Constants and definitions
dict_product_type = {
    'Ações internacionais': 'acoes_int',
    'Ações nacionais': 'acoes_nac',
    'Caixa': 'caixa',
    'Fundos Imobiliários': 'fii',
    'Renda fixa Pós-fixada': 'rf_pos',
    'Renda fixa Pré-fixada': 'rf_pre'
}

color_palette = ['#ffa600', '#ff7c43', '#f95d6a',
                 '#d45087', '#a05195', '#665191', '#2f4b7c', '#003f5c']

df_portfolio = pd.read_csv('portfolio.csv').dropna()
df_target = pd.read_csv('target_prices.csv').dropna()

# Functions
# Calculate stock score, margin and variation on the last month


def calculate_stock_score(ticker, target_price):
    stock = Ticker(ticker.upper() + '.SA')
    prices = stock.history(period="6mo")["close"]
    margin = (1 - prices[-1]/target_price)

    n = len(prices)

    price_0 = prices[-1]
    price_1 = np.mean(prices[int(2*n/3):int(11*n/12):])
    price_2 = np.mean(prices[int(n/3):int(2*n/3)])
    price_3 = np.mean(prices[0:int(n/3)])

    var_1 = (price_0 - price_1)/price_1
    var_2 = (price_0 - price_2)/price_2
    var_3 = (price_0 - price_3)/price_3

    score = (1+margin)**2 * (1-1.5*var_1) * (1-1.3*var_2) * (1-1*var_3)

    return [score, margin, var_1]


def get_ticker(row):
    aux = row.split('-')
    aux = aux[0].strip()
    return aux


def get_type(row):
    if row == 'Renda Fixa Pós':
        return 'rf_pos'
    if row == 'Renda Fixa Pré':
        return 'rf_pre'
    if row == 'BDR - Brazilian Depositary Receipt':
        return 'acoes_int'
    if row == 'Ação':
        return 'acoes_nac'
    if row == 'Fundo Imobiliário':
        return 'fii'
    if row == 'Conta Corrente':
        return 'caixa'
    return 'multi'


def invest_cleaning(df_raw):
    df_raw['type'] = df_raw['Classe do Ativo'].apply(get_type)
    df_raw['ticker'] = np.where(df_raw['type'].isin(
        ['acoes_int', 'acoes_nac', 'fii']), df_raw['Produto'].apply(get_ticker), None)
    df_raw = df_raw.rename(columns={'Valor aplicado': 'value_applied',
                                    'Saldo bruto': 'gross_balance',
                                    'Rentabilidade (%)': 'profit',
                                    'Participação na carteira (%)': 'percentage'})

    df = df_raw[['ticker', 'type', 'value_applied',
                 'gross_balance', 'profit', 'percentage']]
    numeric = ['value_applied', 'gross_balance', 'profit', 'percentage']
    for column in numeric:
        df.loc[:, column] = pd.to_numeric(df[column])
    df = df.loc[df['value_applied'] > 0, :]

    return df


def get_target_price(ticker):
    aux = df_target.loc[df_target['stock'].str.lower() == ticker.lower(), :]
    aux = aux.reset_index(drop=True)
    try:
        return float(aux.loc[0, 'target'])
    except:
        return None


def get_score(df):
    row = df['ticker']
    target_price = get_target_price(row)
    if target_price:
        return calculate_stock_score(row, target_price)
    return [0, 0, 0]


def get_recomendations_from_type(df, type_option):
    product_type = dict_product_type[type_option]

    df_type = df.loc[df['type'] == product_type, :].drop_duplicates(subset=[
                                                                    'ticker'])
    type_results = df_type.apply(get_score, axis=1, result_type='expand')
    df_type = df_type.assign(score=type_results[0],
                             margin=type_results[1],
                             var_1month=type_results[2])
    df_plot = df_type.loc[df_type['score'] != 0, :]
    df_plot = df_plot.sort_values(
        by='score', ascending=False).reset_index(drop=True)

    plot_recomendations(df_plot[0:8], type_option)

    return df_plot


def max_width(prcnt_width: int = 50):
    max_width_str = f"max-width: {prcnt_width}%;"
    st.markdown(f""" 
                <style> 
                .reportview-container .main .block-container{{{max_width_str}}}
                </style>    
                """,
                unsafe_allow_html=True)


def plot_recomendations(df_plot, type_option):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(df_plot['ticker'], df_plot['score'],
            align='center', color=color_palette)
    ax.set_yticks(df_plot['ticker'])
    ax.set_yticklabels(df_plot['ticker'].str.upper(), fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Recomendações de {}'.format(type_option), fontsize=14)

    plt.grid(linewidth=0.2)
    st.pyplot(fig, dpi=1000)
