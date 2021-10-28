# Imports
# Standard
import numpy as np
import pandas as pd
# Data viz
from IPython.display import clear_output
import matplotlib.pyplot as plt
import streamlit as st
# Finances
from yfinance import Ticker
# Statistics
from sklearn.metrics import mean_absolute_percentage_error
from scipy.optimize import minimize
from sklearn.preprocessing import minmax_scale
# Other
import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Constants and definitions
dict_product_type = {
    'Ações internacionais': 'int',
    'Ações nacionais': 'nac',
    'Caixa': 'caixa',
    'Fundos Imobiliários': 'fii',
    'Renda fixa Pós-fixada': 'rf_pos',
    'Renda fixa Pré-fixada': 'rf_pre',
    'Multimercado': 'multi'}

dict_kinvo = {
    'Renda Fixa Pós': 'rf_pos',
    'Renda Fixa Pré': 'rf_pre',
    'BDR - Brazilian Depositary Receipt': 'int',
    'Ação': 'nac',
    'Fundo Imobiliário': 'fii',
    'Conta Corrente': 'caixa'}

color_palette = ['#ffa600', '#ff7c43', '#f95d6a',
                 '#d45087', '#a05195', '#665191', '#2f4b7c', '#003f5c']

stock_types = ['nac', 'int', 'fii']

# Databases
df_target = pd.read_excel(r'target_prices.xlsx', header=1, usecols=['ticker',
                                                                    'ticker_int',
                                                                    'type',
                                                                    'target_price',
                                                                    'currency',
                                                                    'alexandre', 'rafael', 'giovana'])

# Functions


def calculate_stock_score(ticker, target_price, currency):
    if not currency or currency == 'BRL':
        stock = Ticker(ticker.upper() + '.SA')
    else:
        stock = Ticker(ticker.upper())

    prices = stock.history(period="6mo")["Close"].values
    n = len(prices)

    if n > 0:
        price_0 = prices[-1]
        price_1 = np.mean(prices[int(2*n/3):int(11*n/12):])
        price_2 = np.mean(prices[int(n/3):int(2*n/3)])
        price_3 = np.mean(prices[0:int(n/3)])

        var_1 = (price_0 - price_1)/price_1
        var_2 = (price_0 - price_2)/price_2
        var_3 = (price_0 - price_3)/price_3
    else:
        var_1 = None

    try:
        margin = (1 - prices[-1]/target_price)
        score = (1+margin)**2 * (1-1.5*var_1) * (1-1.3*var_2) * (1-1*var_3)
    except:
        margin = None
        score = None

    return [score, margin, var_1]


def get_ticker(row):
    aux = row.split('-')
    aux = aux[0].strip()
    return aux


def get_type(row):
    try:
        return dict_kinvo[row]
    except:
        return 'multi'


def invest_cleaning(df_raw):
    df_raw = df_raw.groupby(
        by=['Produto', 'Classe do Ativo']).sum().reset_index()
    df_raw['type'] = df_raw['Classe do Ativo'].apply(get_type)
    df_raw['ticker'] = np.where(df_raw['type'].isin(
        ['int', 'nac', 'fii']), df_raw['Produto'].apply(get_ticker), None)
    df_raw = df_raw.rename(columns={'Valor aplicado': 'value_applied',
                                    'Saldo bruto': 'gross_balance',
                                    'Rentabilidade (%)': 'profit',
                                    'Participação na carteira (%)': 'percentage'})

    df = df_raw[['ticker', 'type', 'value_applied',
                 'gross_balance', 'profit', 'percentage']]
    df['profit'] = 100 * (df['gross_balance'] / df['value_applied'] - 1)
    numeric = ['value_applied', 'gross_balance', 'profit', 'percentage']
    for column in numeric:
        df.loc[:, column] = pd.to_numeric(df[column])
    df = df.loc[df['value_applied'] > 0, :]

    return df


def get_target_price(ticker):
    aux = df_target.loc[df_target['ticker'].str.lower() == ticker.lower(), :]
    aux = aux.reset_index(drop=True)
    try:
        return float(aux.loc[0, 'target_price']), str(aux.loc[0, 'currency'])
    except:
        return None, None


def get_score(df):
    ticker = df['ticker']
    try:
        ticker_int = df_target.loc[df_target['ticker']
                                   == ticker, 'ticker_int'].values[0]
    except:
        ticker_int = None
    target_price, currency = get_target_price(ticker)
    if not pd.isnull(ticker_int):
        return calculate_stock_score(ticker_int, target_price, currency)
    return calculate_stock_score(ticker, target_price, currency)


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


def max_width(prcnt_width: int = 60):
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

    df_show = df_plot[['ticker', 'score', 'margin', 'var_1month']].rename(
        columns={'ticker': 'Ticker', 'score': 'Score', 'margin': 'Margin (%)', 'var_1month': '1 month var (%)'})
    df_show['Margin (%)'] = df_show['Margin (%)'].apply(
        lambda x: "{:,.2f}%".format((100*x)))
    df_show['1 month var (%)'] = df_show['1 month var (%)'].apply(
        lambda x: "{:,.2f}%".format((100*x)))
    st.table(df_show)


def calculate_mape(y_true, y_pred, x, n_commends):
    clear_output(wait=True)
    # print(*np.round(x, 2), sep='\n')
    aux = np.pad(x, (0, len(y_true) - n_commends),
                 mode='constant', constant_values=0)
    mape = mean_absolute_percentage_error(y_true, y_pred + aux)
    # print('MAPE: {:.1f}%'.format(100*mape))
    return mape


def get_commend_values(df_clean, df_portfolio, cash, investor, n_commends):
    df_aux = df_portfolio[df_portfolio['investor_type']
                          == investor].drop(columns=['investor_type'])
    df_inv = df_aux.T.reset_index().rename(
        columns={'index': 'type', df_aux.index.values[0]: 'ideal_percentage'})
    df_inv = df_inv[df_inv['ideal_percentage'] > 0].merge(df_clean.drop(
        columns=['profit']).groupby('type').sum().reset_index(), on='type', how='left').fillna(0)
    df_inv['ideal_balance'] = df_inv['ideal_percentage'] / \
        100 * (df_inv['gross_balance'].sum() + cash)
    df_inv['difference'] = df_inv['ideal_balance'] - df_inv['gross_balance']
    df_inv = df_inv.sort_values(
        by='difference', ascending=False).reset_index(drop=True)

    # Minimization
    def fun(x): return calculate_mape(
        df_inv['ideal_balance'].values, df_inv['gross_balance'].values, x, n_commends)
    cons = ({'type': 'eq', 'fun': lambda x: np.array(x).sum() - cash})
    bnds = [[0, cash] for i in range((n_commends))]

    x0 = [cash/n_commends for i in range((n_commends))]
    res = minimize(fun, x0, method='trust-constr',
                   bounds=bnds, constraints=cons)

    df_inv.loc[0:n_commends-1, 'commend_value'] = np.round(res.x)
    df_inv['variation_percentage'] = 100 * \
        np.round(df_inv['commend_value'] / df_inv['ideal_balance'].sum(), 3)
    df_inv['type_name'] = df_inv['type'].apply(lambda x: list(
        dict_product_type.keys())[list(dict_product_type.values()).index(x)])
    df_commend = df_inv[df_inv['commend_value'] > 0].reset_index(drop=True)

    cols = st.columns(len(df_commend))
    for i, col in enumerate(cols):

        with col:
            st.metric(df_commend.loc[i, 'type_name'], 'R${:,.0f}'.format(
                df_commend.loc[i, 'commend_value']), '{:.2f}%'.format(df_commend.loc[i, 'variation_percentage']))

    return df_commend


def get_stocks_recommendation_from_type(df_sell, investor, commend_value, invest_type, n_stocks):

    df = df_sell.loc[(df_sell['type'] == invest_type)
                     & (df_sell['in_portfolio'] == 1)]
    # df['ticker_int'] = df['ticker_int'].fillna('-')
    df['currency'] = df['currency'].fillna(method='pad')
    type_results = df.apply(get_score,
                            axis=1, result_type='expand')
    df = df.assign(score=type_results[0],
                   margin=type_results[1],
                   var_1month=type_results[2])
    df['percentage_scaled'] = np.where(df['in_portfolio'] == 1, minmax_scale(
        df['percentage'], feature_range=(0, 1)), None)
    df['score_scaled'] = minmax_scale(df['score'], feature_range=(0, 1))
    df['weights'] = (1 - df['percentage_scaled']) * df['score_scaled']
    df['ideal_balance'] = (df_sell.loc[df_sell['type'] == invest_type,
                           'gross_balance'].sum() + commend_value)/len(df)
    df = df.sort_values('weights', ascending=False).reset_index(drop=True)

    # Minimization
    def fun(x): return calculate_mape(
        df['ideal_balance'].values, df['gross_balance'].values, x, n_stocks)
    cons = ({'type': 'eq', 'fun': lambda x: np.array(x).sum() - commend_value})
    bnds = [[0, commend_value] for i in range((n_stocks))]

    x0 = [commend_value for i in range((n_stocks))]
    res = minimize(fun, x0, method='trust-constr',
                   bounds=bnds, constraints=cons)

    df.loc[0:n_stocks-1, 'commend_value'] = np.round(res.x)
    df['variation_percentage'] = 100 * \
        np.round(df['commend_value'] / df['ideal_balance'].sum(), 3)

    df_stocks = df[df['commend_value'] > 0].reset_index(drop=True)

    cols = st.columns(len(df_stocks))
    for i, col in enumerate(cols):
        with col:
            try:
                var = '{:.2f}%'.format(100*df_stocks.loc[i, 'var_1month'])
            except:
                var = '-'
            st.metric(df_stocks.loc[i, 'ticker'], 'R${:,.0f}'.format(
                df_stocks.loc[i, 'commend_value']), var, delta_color='inverse')

    return df_stocks


def show_stock_commends(df_clean, df_commend, investor):
    df_target_clean = df_target[['ticker', 'ticker_int', 'target_price',
                                 'currency', investor]].rename(columns={investor: 'in_portfolio'})
    df_sell = df_target_clean.merge(
        df_clean.loc[df_clean['type'].isin(stock_types)], on=['ticker'], how='outer')
    df_sell['in_portfolio'] = df_sell['in_portfolio'].fillna(0)
    sell_stocks = df_sell.loc[(df_sell['in_portfolio'] == 0) & (
        df_sell['gross_balance'] > 0), 'ticker'].values
    sell_values = df_sell.loc[(df_sell['in_portfolio'] == 0) & (
        df_sell['gross_balance'] > 0), 'gross_balance'].values

    if len(sell_stocks) > 0:
        st.markdown('***')
        st.markdown('**Recomendações de venda**')
        cols = st.columns(len(sell_stocks))
        for i, col in enumerate(cols):
            with col:
                st.metric(sell_stocks[i], 'R${:,.0f}'.format(
                    sell_values[i]))

    types_to_commend = df_commend.loc[df_commend['type'].isin(
        stock_types), 'type'].values
    for invest_type in types_to_commend:
        type_name = list(dict_product_type.keys())[list(
            dict_product_type.values()).index(invest_type)]
        st.markdown('***')
        st.subheader('{}'.format(type_name))
        commend_value = df_commend.loc[df_commend['type']
                                       == invest_type, 'commend_value'].values[0]
        max_n_stocks = int((df_target[(df_target[investor] == 1) & (
            df_target['type'] == invest_type)] != 0).sum(axis=0).values[0])
        n_stocks = st.slider('Deseja recomendação de quantos ativos?'.format(), min_value=1,
                             max_value=max_n_stocks, value=3, step=1)
        df_stocks = get_stocks_recommendation_from_type(
            df_sell, investor, commend_value, invest_type, n_stocks)
