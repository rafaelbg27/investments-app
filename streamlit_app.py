# Imports
# Standard
import numpy as np
import pandas as pd
# Data viz
from IPython.display import clear_output
import streamlit as st
# Other
import warnings
from pandas.core.common import SettingWithCopyWarning
import aux_functions

# Config
aux_functions.max_width()
warnings.filterwarnings(action="ignore", category=RuntimeWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

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
                         sep=None, engine='python', encoding='utf-8-sig')
    df_portfolio = pd.read_csv('portfolio.csv').dropna()
    df_clean = aux_functions.invest_cleaning(df_raw)
    file.close()

    st.markdown('***')
    st.markdown('**Features disponíveis:**')
    best_option = st.checkbox('Melhores recomendações')
    manual_option = st.checkbox('Seleção manual')
    st.markdown('***')

    if best_option:
        portfolio_option = st.selectbox(
            'Qual seu perfil de investidor?', ['-'] + (list(df_portfolio['investor_type'].str.capitalize())))
        cash = st.number_input('Valor do aporte: R$')

        if cash and portfolio_option != '-':

            st.markdown('***')
            max_n_cats = int((df_portfolio[df_portfolio['investor_type'] == portfolio_option.lower()] != 0).sum(
                axis=1).values[0] - 1)
            n_commends = st.slider('Em quantas categorias de investimento você quer distribuir seu aporte?',
                                   min_value=1, max_value=max_n_cats, value=2, step=1)
            st.subheader('Distribuição recomendada do aporte')
            df_commend = aux_functions.get_commend_values(
                df_clean, df_portfolio, cash, portfolio_option.lower(), n_commends)

            aux_functions.show_stock_commends(
                df_clean, df_commend, portfolio_option.lower())

    if manual_option:
        product_option = st.selectbox('Onde você quer investir?', [
                                      'Ações nacionais', 'Ações internacionais', 'Fundos Imobiliários'])

        aux_functions.get_recomendations_from_type(
            df_clean, product_option)
