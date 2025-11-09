import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.exceptions import OptimizationError
import traceback

def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    # força tudo para numérico (mantém datas no índice)
    return df.apply(pd.to_numeric, errors="coerce")

def Otimizacao_Markowitz(
    quantidade_ativos: int,
    peso_maximo: float,
    taxa_livre_risco: float,
    retorno_alvo: float = None,
) -> dict:
    print("\n===============================")
    print(">> Iniciando Otimização Markowitz")
    print(f"Parâmetros: ativos={quantidade_ativos}, peso_max={peso_maximo}, "
          f"taxa_rf={taxa_livre_risco}, retorno_alvo={retorno_alvo}")
    print("===============================")

    try:
        # --- 1. Carregamento das Bases ---
        df_vol = pd.read_csv(
            "Base Cota Mercado.csv", parse_dates=["dt_pregao"], index_col="dt_pregao"
        )
        df_ret = pd.read_csv(
            "Base Cota Ajustada.csv", parse_dates=["dt_pregao"], index_col="dt_pregao"
        )

        print(f"> Base Mercado: {df_vol.shape[0]} linhas x {df_vol.shape[1]} colunas")
        print(f"> Base Ajustada: {df_ret.shape[0]} linhas x {df_ret.shape[1]} colunas")

        # --- 2. Filtro de Período ---
        start_date = "2020-01-01"
        end_date = "2024-12-31"
        df_vol = df_vol.loc[start_date:end_date]
        df_ret = df_ret.loc[start_date:end_date]
        print(f"> Após filtro: {df_vol.shape[0]} datas válidas")

        # --- 3. Sincronização de colunas ---
        common_tickers = df_vol.columns.intersection(df_ret.columns)
        print(f"> Tickers em comum antes da limpeza: {len(common_tickers)}")

        df_vol = df_vol[common_tickers]
        df_ret = df_ret[common_tickers]

        # --- 4. Limpeza de NaNs ---
        thresh_vol = int(0.90 * len(df_vol))
        thresh_ret = int(0.90 * len(df_ret))
        df_vol.dropna(axis=1, thresh=thresh_vol, inplace=True)
        df_ret.dropna(axis=1, thresh=thresh_ret, inplace=True)

        common_tickers_after_na = df_vol.columns.intersection(df_ret.columns)
        print(f"> Após dropna por coluna: {len(common_tickers_after_na)} ativos restantes")

        df_vol = df_vol[common_tickers_after_na].dropna()
        df_ret = df_ret[common_tickers_after_na].dropna()

        common_index = df_vol.index.intersection(df_ret.index)
        df_vol = df_vol.loc[common_index]
        df_ret = df_ret.loc[common_index]

        print(f"> Após dropna por linha: {df_ret.shape[0]} dias válidos, {df_ret.shape[1]} ativos")

        if df_ret.empty or df_vol.empty:
            raise ValueError("As bases ficaram vazias após limpeza — verifique NaNs ou tickers inconsistentes.")

        # --- 5. Cálculo dos Inputs ---
        mu = expected_returns.mean_historical_return(df_ret)
        S = risk_models.CovarianceShrinkage(df_vol).ledoit_wolf()
        print("> Inputs calculados com sucesso (retornos e covariância)")

        # --- 6. Seleção de ativos ---
        if quantidade_ativos >= len(mu):
            print("> Utilizando todos os ativos disponíveis.")
            selected_tickers = mu.index.tolist()
        else:
            sharpe_individual = (mu - taxa_livre_risco) / np.sqrt(np.diag(S))
            selected_tickers = sharpe_individual.nlargest(quantidade_ativos).index.tolist()
            print(f"> Ativos selecionados: {len(selected_tickers)}")

        mu_sel = mu[selected_tickers]
        S_sel = S.loc[selected_tickers, selected_tickers]

        # --- 7. Otimização ---
        ef = EfficientFrontier(mu_sel, S_sel, weight_bounds=(0, peso_maximo))

        try:
            if retorno_alvo:
                ef.efficient_return(target_return=retorno_alvo)
                print("> Otimização feita por retorno alvo")
            else:
                ef.max_sharpe(risk_free_rate=taxa_livre_risco)
                print("> Otimização feita para máximo Sharpe")
        except (OptimizationError, ValueError) as e:
            print(f"> Erro ao usar retorno alvo ({e}). Tentando max_sharpe...")
            ef.max_sharpe(risk_free_rate=taxa_livre_risco)

        # --- 8. Extração de Pesos ---
        pesos = ef.clean_weights(cutoff=1e-5)
        pesos_final = {ticker: w for ticker, w in pesos.items() if w > 0}

        soma_pesos = sum(pesos_final.values())
        print(f"> Quantidade de ativos na carteira: {len(pesos_final)}")
        print(f"> Soma dos pesos: {soma_pesos:.4f}")

        if len(pesos_final) == 0:
            raise ValueError("Nenhum ativo recebeu peso positivo — otimização inválida.")

        return pesos_final

    except Exception as e:
        print("\n[ERRO na Otimização Markowitz]")
        print(e)
        traceback.print_exc()
        print("Retornando carteira vazia.\n")
        return {}
