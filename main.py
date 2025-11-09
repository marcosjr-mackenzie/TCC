import streamlit as st
import pandas as pd
import markowitz
import deepRF as drl
import comparacao


# --- Inicialização de variáveis no session_state ---
for k in (
    "carteira_markowitz",
    "carteira_drl",
    "fig_cumulative",
    "fig_pies",
    "fig_metrics",
    "metrics_df",
):
    st.session_state.setdefault(k, None)


# --- Função auxiliar para formatar a carteira em DataFrame ---
def formatar_carteira_df(carteira_dict: dict) -> pd.DataFrame:
    if not carteira_dict or not isinstance(carteira_dict, dict):
        st.warning("Nenhuma carteira foi gerada ou o resultado está vazio.")
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(carteira_dict, orient="index", columns=["Peso"])
    df = df.reset_index().rename(columns={"index": "Ativo"})
    df = df.sort_values(by="Peso", ascending=False).reset_index(drop=True)
    return df


# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="Otimizador de Carteiras")

st.title("Otimização de Carteiras: Markowitz vs. Deep Reinforcement Learning")
st.markdown(
    "Aplicação desenvolvida para o TCC — comparação entre modelos de otimização de carteiras de fundos imobiliários (FIIs)."
)

# --- Barra Lateral: Parâmetros ---
with st.sidebar:
    st.header("Parâmetros de Otimização")

    quantidade_ativos = st.number_input(
        "1. Número de ativos na carteira",
        min_value=2,
        max_value=50,
        value=10,
        step=1,
        help="Quantidade exata de ativos que devem compor a carteira final.",
    )

    peso_maximo = (
        st.slider(
            "2. Peso máximo por ativo (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            help="Peso máximo permitido para um ativo na carteira.",
        )
        / 100.0
    )

    retorno_alvo = (
        st.number_input(
            "3. Retorno anual alvo (%) — Markowitz",
            min_value=1.0,
            max_value=50.0,
            value=15.0,
            step=0.5,
            help="Retorno anual alvo da carteira (utilizado apenas na otimização de Markowitz).",
        )
        / 100.0
    )

    taxa_livre_risco = (
        st.number_input(
            "4. Taxa Livre de Risco (Selic) Anual (%)",
            min_value=0.0,
            max_value=30.0,
            value=10.5,
            step=0.25,
            help="Taxa usada como base para o cálculo do Índice de Sharpe.",
        )
        / 100.0
    )


# --- Botões de Otimização ---
st.divider()
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Otimização Clássica — Markowitz")
    if st.button("Otimizar com Markowitz", use_container_width=True, type="secondary"):
        with st.spinner("Calculando a carteira ótima com o modelo de Markowitz..."):
            try:
                st.session_state.carteira_markowitz = markowitz.Otimizacao_Markowitz(
                    quantidade_ativos=quantidade_ativos,
                    peso_maximo=peso_maximo,
                    taxa_livre_risco=taxa_livre_risco,
                    retorno_alvo=retorno_alvo,
                )
                st.success("Carteira Markowitz gerada com sucesso!")
            except Exception as e:
                st.error(f"Ocorreu um erro na otimização Markowitz: {e}")

with col2:
    st.subheader("Otimização com DRL")
    if st.button("Otimizar com DRL", use_container_width=True, type="secondary"):
        st.info("O treinamento do modelo de DRL pode levar alguns minutos. Aguarde...")
        with st.spinner("Treinando o agente e otimizando a carteira..."):
            try:
                st.session_state.carteira_drl = drl.otimizacao_deepRF(
                    num_assets=quantidade_ativos,
                    max_weight_per_asset=peso_maximo,
                    risk_free_rate=taxa_livre_risco,
                    target_return=retorno_alvo,
                )
                st.success("Carteira DRL gerada com sucesso!")
            except Exception as e:
                st.error(f"Ocorreu um erro na otimização com DRL: {e}")


# --- Exibição das Carteiras ---
if st.session_state.carteira_markowitz or st.session_state.carteira_drl:
    st.divider()
    st.header("Resultados das Otimizações")

    res_col1, res_col2 = st.columns(2, gap="large")

    # --- Modelo Markowitz ---
    with res_col1:
        if st.session_state.carteira_markowitz:
            st.write("**Carteira - Modelo Markowitz:**")
            df_markowitz = formatar_carteira_df(st.session_state.carteira_markowitz)
            st.dataframe(
                df_markowitz.style.format({"Peso": "{:.2%}"}), use_container_width=True
            )

    # --- Modelo DRL ---
    with res_col2:
        if st.session_state.carteira_drl:
            st.write("**Carteira - Modelo DRL:**")
            df_drl = formatar_carteira_df(st.session_state.carteira_drl)
            st.dataframe(
                df_drl.style.format({"Peso": "{:.2%}"}), use_container_width=True
            )


# --- Seção Comparativa (Backtest) ---
if st.session_state.carteira_markowitz and st.session_state.carteira_drl:
    st.divider()
    st.header("Análise Comparativa de Desempenho (Backtest)")

    if st.button(
        "Comparar Desempenho Histórico das Carteiras",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Executando backtest e gerando gráficos comparativos..."):
            (
                fig_cum,
                fig_pie,
                fig_met,
                metrics,
            ) = comparacao.run_backtest_and_plot(
                st.session_state.carteira_markowitz,
                st.session_state.carteira_drl,
                taxa_livre_risco,
            )

            # Armazena os resultados no estado da sessão
            st.session_state.fig_cumulative = fig_cum
            st.session_state.fig_pies = fig_pie
            st.session_state.fig_metrics = fig_met
            st.session_state.metrics_df = pd.DataFrame(metrics).T

    # --- Exibição dos resultados comparativos ---
    if st.session_state.fig_cumulative is not None:
        st.plotly_chart(st.session_state.fig_cumulative, use_container_width=True)
    if st.session_state.fig_pies is not None:
        st.plotly_chart(st.session_state.fig_pies, use_container_width=True)
    if st.session_state.fig_metrics is not None:
        st.plotly_chart(st.session_state.fig_metrics, use_container_width=True)

    st.subheader("Tabela de Métricas Comparativas")
    df_metrics = st.session_state.metrics_df
    if isinstance(df_metrics, pd.DataFrame) and not df_metrics.empty:
        styled_metrics = df_metrics.style.format(
            {
                "Retorno Anualizado": "{:.2%}",
                "Volatilidade Anualizada": "{:.2%}",
                "Máximo Drawdown": "{:.2%}",
                "Índice de Sharpe": "{:.2f}",
            }
        )
        st.dataframe(styled_metrics, use_container_width=True)
    else:
        st.info(
            "Clique em **Comparar Desempenho Histórico das Carteiras** para gerar as métricas."
        )
