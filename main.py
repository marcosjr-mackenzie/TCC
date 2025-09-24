import streamlit as st
import pandas as pd
import markowitz
import deepRF as drl
import comparacao


def formatar_carteira_df(carteira_dict: dict) -> pd.DataFrame:
    if not carteira_dict or not isinstance(carteira_dict, dict):
        st.warning("Nenhuma carteira foi gerada ou o resultado está vazio.")
        return pd.DataFrame()

    # Converte o dicionário para DataFrame da forma correta
    df = pd.DataFrame.from_dict(carteira_dict, orient="index", columns=["Peso"])
    df = df.reset_index().rename(columns={"index": "Ativo"})

    # Ordena os ativos pelo peso para melhor visualização
    df = df.sort_values(by="Peso", ascending=False).reset_index(drop=True)
    return df


# --- Configuração da Página ---

st.set_page_config(layout="wide", page_title="Otimizador de Carteiras")

st.title("Otimização de Carteiras: Markowitz vs. Deep Reinforcement Learning")
st.markdown(
    "Aplicação para o TCC de comparação entre modelos de otimização de carteiras de fundos imobiliários (FIIs)."
)

# --- Barra Lateral (Sidebar) com os Inputs ---
with st.sidebar:
    st.header("Parâmetros de Otimização")

    # st.info(f"Base de dados carregada com {dados['TICKER'].nunique()} ativos.")

    # 1. Número de ativos
    quantidade_ativos = st.number_input(
        "1. Número de ativos na carteira",
        min_value=2,
        max_value=50,
        value=10,
        step=1,
        help="Quantidade exata de ativos que devem compor a carteira final.",
    )

    # 2. Peso máximo por ativo
    peso_maximo = (
        st.slider(
            "2. Peso máximo por ativo (%)",
            min_value=5,
            max_value=50,
            value=20,
            step=1,
            help="O peso máximo que um único ativo pode ter na carteira.",
        )
        / 100.0
    )

    # 3. Retorno alvo
    retorno_alvo = (
        st.number_input(
            "3. Retorno anual alvo (%) - (Markowitz)",
            min_value=1.0,
            max_value=50.0,
            value=15.0,
            step=0.5,
            help="Define um retorno alvo para a otimização de Markowitz. Se não for atingível, o modelo focará em maximizar o Sharpe.",
        )
        / 100.0
    )

    # 4. Taxa Livre de Risco
    taxa_livre_risco = (
        st.number_input(
            "4. Taxa Livre de Risco (Selic) Anual (%)",
            min_value=0.0,
            max_value=30.0,
            value=10.5,
            step=0.25,
            help="Usada como base para o cálculo do Índice de Sharpe.",
        )
        / 100.0
    )

# --- Corpo Principal da Aplicação ---
st.divider()
col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("Otimização Clássica")
    if st.button("Otimizar com Markowitz", use_container_width=True, type="secondary"):
        with st.spinner("Calculando a carteira ótima por Markowitz..."):
            try:
                st.session_state.carteira_markowitz = markowitz.Otimizacao_Markowitz(
                    quantidade_ativos=quantidade_ativos,
                    peso_maximo=peso_maximo,
                    taxa_livre_risco=taxa_livre_risco,
                    retorno_alvo=retorno_alvo,
                )
            except Exception as e:
                st.error(f"Ocorreu um erro na otimização Markowitz: {e}")

with col2:
    st.subheader("Otimização com IA")
    if st.button("Otimizar com DRL", use_container_width=True, type="secondary"):
        # Mensagem de aviso sobre o tempo de execução
        st.info(
            "O treinamento do modelo de DRL pode levar alguns minutos. Por favor, aguarde."
        )
        with st.spinner("Treinando agente de IA e otimizando a carteira..."):
            try:
                st.session_state.carteira_drl = drl.otimizacao_deepRF(
                    num_assets=quantidade_ativos,
                    max_weight_per_asset=peso_maximo,
                    risk_free_rate=taxa_livre_risco,
                    target_return=retorno_alvo,
                )
                print("teste")
            except Exception as e:
                st.error(f"Ocorreu um erro na otimização com DRL: {e}")

# --- Seção para exibir os resultados ---
if "carteira_markowitz" in st.session_state or "carteira_drl" in st.session_state:
    st.divider()
    st.header("Resultados das Otimizações")

    res_col1, res_col2 = st.columns(2)

    with res_col1:
        if (
            "carteira_markowitz" in st.session_state
            and st.session_state.carteira_markowitz
        ):
            st.write("Carteira - Modelo Markowitz:")
            df_markowitz = formatar_carteira_df(st.session_state.carteira_markowitz)
            st.dataframe(
                df_markowitz.style.format({"Peso": "{:.2%}"}), use_container_width=True
            )

    with res_col2:
        if "carteira_drl" in st.session_state and st.session_state.carteira_drl:
            st.write("Carteira - Modelo DRL:")
            df_ml = formatar_carteira_df(st.session_state.carteira_drl)
            st.dataframe(
                df_ml.style.format({"Peso": "{:.2%}"}), use_container_width=True
            )

# --- 4. Seção para Exibir os Resultados ---
# Esta seção é renderizada sempre. Se uma carteira existir no st.session_state, ela será exibida.
# Isso garante que os resultados permaneçam visíveis.

if "carteira_markowitz" in st.session_state or "carteira_drl" in st.session_state:
    st.divider()
    st.header("Resultados das Otimizações")

    res_col1, res_col2 = st.columns(2, gap="large")

    with res_col1:
        if "carteira_markowitz" in st.session_state:
            st.write("Carteira - Modelo Markowitz:")
            df_markowitz = formatar_carteira_df(st.session_state.carteira_markowitz)
            st.dataframe(
                df_markowitz.style.format({"Peso": "{:.2%}"}), use_container_width=True
            )

    with res_col2:
        if "carteira_drl" in st.session_state:
            st.write("Carteira - Modelo DRL:")
            df_ml = formatar_carteira_df(st.session_state.carteira_drl)
            st.dataframe(
                df_ml.style.format({"Peso": "{:.2%}"}), use_container_width=True
            )

# --- Seção de Comparação (Integrada com o novo módulo) ---
if "carteira_markowitz" in st.session_state and "carteira_drl" in st.session_state:
    st.divider()
    st.header("Análise Comparativa de Desempenho (Backtest)")

    if st.button(
        "Comparar Desempenho Histórico das Carteiras",
        type="primary",
        use_container_width=True,
    ):
        # Garante que as carteiras não estão vazias antes de prosseguir
        if st.session_state.carteira_markowitz and st.session_state.carteira_drl:
            with st.spinner("Realizando backtest e gerando gráficos..."):
                fig_cum, fig_pie, fig_met, metrics = comparacao.run_backtest_and_plot(
                    st.session_state.carteira_markowitz,
                    st.session_state.carteira_drl,
                    taxa_livre_risco,
                )
                # Salva os resultados no estado da sessão para que não desapareçam
                st.session_state.fig_cumulative = fig_cum
                st.session_state.fig_pies = fig_pie
                st.session_state.fig_metrics = fig_met
                st.session_state.metrics_df = pd.DataFrame(metrics).T
        else:
            st.warning(
                "É necessário que ambas as carteiras tenham sido geradas com sucesso para a comparação."
            )

    # Exibe os gráficos e a tabela se eles existirem no estado da sessão
    if "fig_cumulative" in st.session_state:
        st.plotly_chart(st.session_state.fig_cumulative, use_container_width=True)
        st.plotly_chart(st.session_state.fig_pies, use_container_width=True)
        st.plotly_chart(st.session_state.fig_metrics, use_container_width=True)

        st.subheader("Tabela de Métricas Comparativas")
        # Formata o dataframe de métricas para melhor visualização
        styled_metrics = st.session_state.metrics_df.style.format(
            {
                "Retorno Anualizado": "{:.2%}",
                "Volatilidade Anualizada": "{:.2%}",
                "Máximo Drawdown": "{:.2%}",
                "Índice de Sharpe": "{:.2f}",
            }
        )
        st.dataframe(styled_metrics, use_container_width=True)
