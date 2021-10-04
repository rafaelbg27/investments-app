# Imports
# Standard
import aux_functions
import numpy as np
import pandas as pd
# Data viz
import matplotlib.pyplot as plt
# Finances
from yahooquery import Ticker
# Other
import streamlit as st
import warnings
from pandas.core.common import SettingWithCopyWarning


aux_functions.max_width()
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Constants and definitions
# dict_product_type = {
#     'Ações internacionais': 'acoes_int',
#     'Ações nacionais': 'acoes_nac',
#     'Caixa': 'caixa',
#     'Fundos Imobiliários': 'fii',
#     'Renda fixa Pós-fixada': 'rf_pos',
#     'Renda fixa Pré-fixada': 'rf_pre'
# }

# color_palette = ['#ffa600', '#ff7c43', '#f95d6a',
#                  '#d45087', '#a05195', '#665191', '#2f4b7c', '#003f5c']

# Main
st.title('Oportunidades de investimento')
st.subheader(
    'Vamos encontrar os melhores investimentos para você usando apenas o CSV gerado pelo Kinvo!')
st.markdown('***')

# File upload
file = st.file_uploader(
    'Faça upload do arquivo gerado pelo Kinvo em formato CSV.', type=['csv'])

if file:
    df_raw = pd.read_csv(file, thousands='.', decimal=',',
                         sep=None, engine='python')
    file.close()

    df_portfolio = pd.read_csv('portfolio.csv').dropna()
    df_target = pd.read_csv('target_prices.csv').dropna()

    st.markdown('***')
    st.markdown('**Features disponíveis:**')
    best_option = st.checkbox('Melhores recomendações')
    st.markdown('***')

    if best_option:
        st.text('Essa feature ainda está em desenvolvimento :(')
        portfolio_option = st.selectbox(
            'Qual seu perfil de investidor?', df_portfolio['investor_type'].str.capitalize())

    else:
        product_option = st.selectbox('Onde você quer investir?', [
                                      'Ações nacionais', 'Ações internacionais', 'Fundos Imobiliários'])
        df_clean = aux_functions.invest_cleaning(df_raw)
        aux_functions.get_recomendations_from_type(
            df_clean, product_option)
