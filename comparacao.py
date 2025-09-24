import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def calculate_performance_metrics(daily_returns, risk_free_rate=0.105):
    """Calcula as principais métricas de desempenho de uma série de retornos diários."""
    if daily_returns.empty:
        return {
            "Retorno Anualizado": 0,
            "Volatilidade Anualizada": 0,
            "Índice de Sharpe": 0,
            "Máximo Drawdown": 0,
        }

    # Constante para anualização
    trading_days = 252

    # Retorno Anualizado
    mean_daily_return = daily_returns.mean()
    annualized_return = (1 + mean_daily_return) ** trading_days - 1

    # Volatilidade Anualizada
    annualized_volatility = daily_returns.std() * np.sqrt(trading_days)
    if annualized_volatility == 0:
        annualized_volatility = 1e-9  # Evitar divisão por zero

    # Índice de Sharpe
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Máximo Drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    return {
        "Retorno Anualizado": annualized_return,
        "Volatilidade Anualizada": annualized_volatility,
        "Índice de Sharpe": sharpe_ratio,
        "Máximo Drawdown": max_drawdown,
    }


def run_backtest_and_plot(
    carteira_markowitz: dict, carteira_drl: dict, taxa_livre_risco: float
):
    """
    Executa o backtest para duas carteiras e gera gráficos comparativos.
    """
    try:
        # --- 1. Carregamento e Preparação dos Dados ---
        # Usamos a cota ajustada para o backtest, pois reflete o retorno total
        df_returns = pd.read_csv(
            "Base Cota Ajustada.csv", parse_dates=["dt_pregao"], index_col="dt_pregao"
        )
        df_returns = df_returns.pct_change().dropna()  # Calcula os retornos diários

        # Define o período do backtest (e.g., a partir de 2022 para testar "fora da amostra")
        df_backtest = df_returns.loc["2022-01-01":]

        portfolios = {"Markowitz": carteira_markowitz, "DRL (IA)": carteira_drl}
        df_cumulative_returns = pd.DataFrame(index=df_backtest.index)
        all_metrics = {}

        # --- 2. Cálculo do Desempenho de Cada Carteira ---
        for name, weights_dict in portfolios.items():
            tickers = list(weights_dict.keys())
            weights = np.array(list(weights_dict.values()))

            # Garante que temos os dados de retorno para todos os ativos da carteira
            if not all(ticker in df_backtest.columns for ticker in tickers):
                print(
                    f"Aviso: Nem todos os ativos da carteira '{name}' estão disponíveis no período de backtest."
                )
                continue

            portfolio_returns = df_backtest[tickers].dot(weights)
            df_cumulative_returns[name] = (1 + portfolio_returns).cumprod() * 100

            # Calcula as métricas de desempenho
            all_metrics[name] = calculate_performance_metrics(
                portfolio_returns, taxa_livre_risco
            )

        # --- 3. Geração dos Gráficos Comparativos ---

        # Gráfico 1: Rentabilidade Acumulada
        fig_cumulative = go.Figure()
        for col in df_cumulative_returns.columns:
            fig_cumulative.add_trace(
                go.Scatter(
                    x=df_cumulative_returns.index,
                    y=df_cumulative_returns[col],
                    mode="lines",
                    name=col,
                )
            )
        fig_cumulative.update_layout(
            title="<b>Rentabilidade Acumulada (R$ 100 Iniciais)</b>",
            xaxis_title="Data",
            yaxis_title="Valor da Carteira (R$)",
            legend_title="Carteira",
            template="plotly_white",
        )

        # Gráfico 2: Alocação de Ativos (Pizza)
        fig_pies = make_subplots(
            rows=1,
            cols=2,
            specs=[[{"type": "domain"}, {"type": "domain"}]],
            subplot_titles=("<b>Alocação Markowitz</b>", "<b>Alocação DRL (IA)</b>"),
        )
        # Tenta extrair as carteiras, tratando o caso de uma delas não existir
        mkw_labels = list(carteira_markowitz.keys())
        mkw_values = list(carteira_markowitz.values())
        drl_labels = list(carteira_drl.keys())
        drl_values = list(carteira_drl.values())

        fig_pies.add_trace(
            go.Pie(labels=mkw_labels, values=mkw_values, name="Markowitz"), 1, 1
        )
        fig_pies.add_trace(
            go.Pie(labels=drl_labels, values=drl_values, name="DRL"), 1, 2
        )
        fig_pies.update_traces(hole=0.4, hoverinfo="label+percent+name")
        fig_pies.update_layout(title_text="<b>Composição das Carteiras Otimizadas</b>")

        # Gráfico 3: Comparação de Métricas (Barras)
        df_metrics = pd.DataFrame(all_metrics).T
        fig_metrics = go.Figure()
        colors = {
            "Retorno Anualizado": "green",
            "Volatilidade Anualizada": "red",
            "Índice de Sharpe": "blue",
            "Máximo Drawdown": "orange",
        }

        for metric in df_metrics.columns:
            # Formatação especial para percentuais
            if "Retorno" in metric or "Volatilidade" in metric or "Drawdown" in metric:
                text_template = "%{y:.2%}"
            else:  # Formatação para números decimais (Sharpe)
                text_template = "%{y:.2f}"

            fig_metrics.add_trace(
                go.Bar(
                    name=metric,
                    x=df_metrics.index,
                    y=df_metrics[metric],
                    text=df_metrics[metric],
                    textposition="auto",
                    texttemplate=text_template,
                )
            )

        fig_metrics.update_layout(
            barmode="group",
            title_text="<b>Métricas de Desempenho Comparativas</b>",
            yaxis_title="Valor",
            xaxis_title="Modelo",
            template="plotly_white",
        )

        return fig_cumulative, fig_pies, fig_metrics, all_metrics

    except Exception as e:
        print(f"Ocorreu um erro ao gerar a comparação: {e}")
        return None, None, None, None
