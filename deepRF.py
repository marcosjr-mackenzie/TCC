import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from pypfopt import risk_models, expected_returns


# --- O Ambiente de Simulação (Gymnasium) ---
class PortfolioEnv(gym.Env):
    """
    Ambiente customizado para simulação de um portfólio de investimentos.
    """

    def __init__(
        self,
        df_prices,
        max_weight,
        target_return=None,
        risk_free_rate=0.10,
        window_size=30,
    ):
        super(PortfolioEnv, self).__init__()

        self.df = df_prices
        self.window_size = window_size
        self.num_assets = df_prices.shape[1]
        self.max_weight = max_weight
        self.target_return = target_return
        self.risk_free_rate = risk_free_rate

        # Espaço de Ações: um vetor contínuo com os pesos de cada ativo
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_assets,), dtype=np.float32
        )

        # --- CORREÇÃO PRINCIPAL: Espaço de Observação ---
        # A observação agora é um vetor plano (1D) contendo:
        # 1. O histórico de preços normalizados (window_size * num_assets)
        # 2. Os pesos atuais da carteira (num_assets)
        obs_shape = (self.window_size * self.num_assets) + self.num_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.weights = np.full(self.num_assets, 1 / self.num_assets)
        self.portfolio_returns = []
        return self._next_observation(), {}

    def step(self, action):
        # Ação é a nova alocação de pesos (normalizamos com softmax)
        self.weights = np.exp(action) / np.sum(np.exp(action))

        self.current_step += 1
        terminated = self.current_step >= len(self.df)

        # Pequena correção no cálculo do retorno para evitar erro de índice
        if not terminated:
            price_change = (
                self.df.iloc[self.current_step] / self.df.iloc[self.current_step - 1]
            )
            portfolio_return = np.dot(price_change - 1, self.weights)
            self.portfolio_returns.append(portfolio_return)
            reward = self._calculate_reward(portfolio_return)
        else:
            reward = 0

        obs = self._next_observation()
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _next_observation(self):
        end = self.current_step
        start = end - self.window_size

        # Pega o histórico de preços
        price_history = self.df.iloc[start:end].values

        # Normaliza os preços pelo último dia para focar na variação percentual
        normalized_prices = price_history / price_history[-1]

        # Achata a matriz de preços em um vetor 1D
        flat_prices = normalized_prices.flatten()

        # Concatena os preços achatados com os pesos atuais para formar o estado final
        obs = np.concatenate([flat_prices, self.weights])
        return obs.astype(np.float32)

    def _calculate_reward(self, portfolio_return):
        weight_penalty = 0
        if np.any(self.weights > self.max_weight):
            excesso = np.max(self.weights) - self.max_weight
            weight_penalty = -10 * excesso

        if self.target_return is not None:
            daily_target = (1 + self.target_return) ** (1 / 252) - 1
            reward = -((portfolio_return - daily_target) ** 2) * 1e3
        else:
            if len(self.portfolio_returns) < 2:
                return 0
            mean_return = np.mean(self.portfolio_returns)
            std_return = np.std(self.portfolio_returns)
            if std_return == 0:
                return 0
            daily_risk_free = (1 + self.risk_free_rate) ** (1 / 252) - 1
            sharpe = (mean_return - daily_risk_free) / std_return
            reward = sharpe

        return reward + weight_penalty


# --- Função Principal de Otimização ---
def otimizacao_deepRF(
    num_assets: int,
    max_weight_per_asset: float,
    risk_free_rate: float,
    target_return,
    training_timesteps: int = 1000,
) -> dict:
    try:
        df_ret = pd.read_csv(
            "Base Cota Ajustada.csv", parse_dates=["dt_pregao"], index_col="dt_pregao"
        )
        df_vol = pd.read_csv(
            "Base Cota Mercado.csv", parse_dates=["dt_pregao"], index_col="dt_pregao"
        )

        start_date, end_date = "2020-01-01", "2024-12-31"
        df_ret, df_vol = (
            df_ret.loc[start_date:end_date],
            df_vol.loc[start_date:end_date],
        )
        common_tickers = df_ret.columns.intersection(df_vol.columns)
        df_ret, df_vol = df_ret[common_tickers], df_vol[common_tickers]
        df_ret.dropna(axis=1, thresh=int(0.90 * len(df_ret)), inplace=True)
        df_vol.dropna(axis=1, thresh=int(0.90 * len(df_vol)), inplace=True)
        common_tickers_after_na = df_ret.columns.intersection(df_vol.columns)
        df_ret = df_ret[common_tickers_after_na].dropna()
        df_vol = df_vol[common_tickers_after_na].dropna()
        common_index = df_ret.index.intersection(df_vol.index)
        df_ret, df_vol = df_ret.loc[common_index], df_vol.loc[common_index]

        if num_assets >= len(df_ret.columns):
            selected_tickers = df_ret.columns.tolist()
        else:
            mu = expected_returns.mean_historical_return(df_ret)
            S = risk_models.CovarianceShrinkage(df_vol).ledoit_wolf()
            sharpe_individual = (mu - risk_free_rate) / np.sqrt(np.diag(S))
            selected_tickers = sharpe_individual.nlargest(num_assets).index.tolist()

        print(f"Ativos selecionados para DRL: {selected_tickers}")
        df_final_for_env = df_ret[selected_tickers]

        env = PortfolioEnv(
            df_prices=df_final_for_env,
            max_weight=max_weight_per_asset,
            target_return=target_return,
            risk_free_rate=risk_free_rate,
        )

        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=training_timesteps)

        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        final_weights_raw = np.exp(action) / np.sum(np.exp(action))
        final_portfolio = dict(zip(selected_tickers, final_weights_raw))

        return final_portfolio

    except Exception as e:
        import traceback

        print(f"Ocorreu um erro inesperado durante a otimização DRL: {e}")
        traceback.print_exc()
        return {}


# --- Bloco de Teste ---
if __name__ == "__main__":
    carteira = otimizacao_deepRF(
        num_assets=10,
        max_weight_per_asset=0.20,
        risk_free_rate=0.105,
        target_return=None,
        training_timesteps=200,
    )

    if carteira:
        print("\n--- Carteira Otimizada com DRL ---")
        total = 0
        for ativo, peso in sorted(
            carteira.items(), key=lambda item: item[1], reverse=True
        ):
            print(f"{ativo}: {peso:.2%}")
            total += peso
        print("---------------------------------")
        print(f"Peso Total: {total:.2%}")
