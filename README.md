# TCC
Otimização de Carteiras de Fundos com métodos de Markowitz e de Aprendizado com Reforço

## Resumo
Este projeto apresenta uma aplicação desenvolvida em Python e Streamlit para comparar dois modelos de otimização de carteiras de fundos imobiliários (FIIs): o clássico modelo de Markowitz e um modelo baseado em Aprendizado por Reforço Profundo (Deep Reinforcement Learning – DRL). A proposta integra fundamentos da Teoria Moderna do Portfólio com técnicas de Inteligência Artificial, avaliando como abordagens adaptativas podem aprimorar a eficiência na alocação de ativos do mercado brasileiro.

A ferramenta permite que o usuário defina parâmetros como número de ativos, peso máximo por ativo, taxa livre de risco, retorno-alvo e timesteps de treinamento, executando a otimização em tempo real e exibindo os resultados em forma de tabelas e gráficos interativos. Além da comparação quantitativa, a aplicação oferece uma análise visual das carteiras, métricas de desempenho e rentabilidade acumulada, permitindo explorar de forma prática e intuitiva os conceitos de otimização de portfólio.

## Dados Utilizados
Os dados utilizados no projeto referem-se a fundos de investimento imobiliário (FIIs) listados na B3 – Brasil, Bolsa, Balcão, abrangendo o período de 2020 a 2024. Foram utilizadas duas bases principais:
- Base Cota Ajustada.csv → contém os valores de cotas ajustados por rendimentos e desdobramentos, utilizada para o cálculo dos retornos diários;
- Base Cota Mercado.csv → contém as cotações de fechamento de mercado, utilizada para o cálculo das covariâncias e volatilidades.
Para garantir realismo e representatividade, foram selecionados apenas fundos com média de liquidez diária superior a R$ 1 milhão, de modo a refletir condições de investimento plausíveis no mercado brasileiro.
