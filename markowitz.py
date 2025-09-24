import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError


def Otimizacao_Markowitz(
    quantidade_ativos: int,
    peso_maximo: float,
    taxa_livre_risco: float,
    retorno_alvo: float = None,
) -> dict:
    try:
        # --- 1. Carregamento e Preparação dos Dados ---
        # Utiliza 'Base Cota Mercado.csv' para o cálculo da volatilidade (risco)
        df_vol = pd.read_csv(
            "Base Cota Mercado.csv", parse_dates=["dt_pregao"], index_col="dt_pregao"
        )
        # Utiliza 'Base Cota Ajustada.csv' para o cálculo dos retornos
        df_ret = pd.read_csv(
            "Base Cota Ajustada.csv", parse_dates=["dt_pregao"], index_col="dt_pregao"
        )

        # Filtra para o período de análise
        start_date = "2020-01-01"
        end_date = "2024-12-31"
        df_vol = df_vol.loc[start_date:end_date]
        df_ret = df_ret.loc[start_date:end_date]

        # Sincroniza as colunas entre os dois dataframes
        common_tickers = df_vol.columns.intersection(df_ret.columns)
        df_vol = df_vol[common_tickers]
        df_ret = df_ret[common_tickers]

        # Limpeza de dados: remove ativos com muitos dados faltantes e depois dias com algum dado faltante
        df_vol.dropna(axis=1, thresh=int(0.90 * len(df_vol)), inplace=True)
        df_ret.dropna(axis=1, thresh=int(0.90 * len(df_ret)), inplace=True)

        common_tickers_after_na = df_vol.columns.intersection(df_ret.columns)
        df_vol = df_vol[common_tickers_after_na].dropna()
        df_ret = df_ret[common_tickers_after_na].dropna()

        # Sincroniza os índices após a remoção de NaNs
        common_index = df_vol.index.intersection(df_ret.index)
        df_vol = df_vol.loc[common_index]
        df_ret = df_ret.loc[common_index]

        # --- 2. Cálculo dos Inputs para a Otimização (Universo Completo) ---
        # Verifica a quantidade de ativos disponíveis
        mu = expected_returns.mean_historical_return(df_ret)
        S = risk_models.CovarianceShrinkage(df_vol).ledoit_wolf()

        # --- 3. Pré-Seleção dos Melhores Ativos ---
        if quantidade_ativos >= len(mu):
            print(
                f"Aviso: O número de ativos solicitados ({quantidade_ativos}) é maior ou igual ao "
                f"número de ativos disponíveis ({len(mu)}). Utilizando todos os ativos."
            )
            selected_tickers = mu.index.tolist()
        else:
            # Calcula o Índice de Sharpe individual para cada ativo
            sharpe_individual = (mu - taxa_livre_risco) / np.sqrt(np.diag(S))

            # Seleciona os N ativos com os maiores Índices de Sharpe
            selected_tickers = sharpe_individual.nlargest(
                quantidade_ativos
            ).index.tolist()

        # Filtra os inputs de retorno e covariância para conter apenas os ativos selecionados
        mu_selected = mu[selected_tickers]
        S_selected = S.loc[selected_tickers, selected_tickers]

        # --- 4. Otimização da Carteira (com Ativos Pré-Selecionados) ---
        # A otimização agora ocorre apenas no subconjunto de ativos.
        ef = EfficientFrontier(mu_selected, S_selected, weight_bounds=(0, peso_maximo))

        if retorno_alvo:
            try:
                ef.efficient_return(target_return=retorno_alvo)
            except (OptimizationError, ValueError) as e:
                print(
                    f"Não foi possível otimizar para o retorno alvo de {retorno_alvo:.2%}. Erro: {e}. "
                    "Otimizando para o máximo Índice de Sharpe."
                )
                ef.max_sharpe(risk_free_rate=taxa_livre_risco)
        else:
            ef.max_sharpe(risk_free_rate=taxa_livre_risco)

        # --- 5. Extração dos Resultados ---
        # Usamos cutoff=0 para garantir que mesmo pesos muito pequenos sejam retornados,
        # mantendo a quantidade de ativos no dicionário final.
        pesos = ef.clean_weights(cutoff=1e-5)

        # Filtra ativos com peso zero para retornar apenas os que compõem a carteira
        pesos_final = {ticker: weight for ticker, weight in pesos.items() if weight > 0}

        return pesos_final

    except Exception as e:
        print(f"Ocorreu um erro inesperado durante a otimização: {e}")
        return {}


# --- TESTES ---
if __name__ == "__main__":
    carteira = Otimizacao_Markowitz(
        quantidade_ativos=6,
        peso_maximo=0.20,
        taxa_livre_risco=0.10,
        retorno_alvo=None,  # Se None, otimiza para max Sharpe
    )

    if carteira:
        print("\n--- Composição Final da Carteira ---")
        total_weight = 0
        for ativo, peso in carteira.items():
            print(f"{ativo}: {peso:.2%}")
            total_weight += peso
        print("----------------------------------")
        print(f"Número de Ativos na Carteira: {len(carteira)}")
        print(f"Peso Total na Carteira: {total_weight:.2%}")
