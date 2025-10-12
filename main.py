import numpy as np
import random

from battery import Battery
from market import Market
from plotter import plot_actions

# Market and battery
initial_soc = 5
capacity = 10
battery = Battery(soc=initial_soc, capacity=capacity, power=5)
market = Market(pattern="square", duration_hours=24)
market.plot()

# Helper functions
def state_to_index(soc, price):
    soc_bin = int(round(soc))
    price_level = 1 if price > 0 else 0
    return soc_bin + price_level * (capacity + 1)

# Q learning setup
n_states = (capacity + 1) * 2
actions = [0, 1]
Q = np.zeros((n_states, len(actions)))
alpha = 0.05
gamma = 0.95
epsilon = 0.2
n_episodes = 500
timestep = 15

def step(time_index, action):
    price = market.prices[time_index]
    reward = 0.0
    if action == 0:  # discharge
        delivered = -battery.discharge(duration=timestep)
        reward = price * delivered
    else:            # charge
        stored = battery.charge(duration=timestep)
        reward = -price * stored
    next_index = time_index + 1
    return reward, next_index

# --- Training ---
for ep in range(n_episodes):
    battery.reset(soc=initial_soc)
    t = 0
    s = state_to_index(battery.get_soc(), market.prices[t])
    while t < market.num_steps:
        if random.random() < epsilon:
            a = random.choice([0, 1])
        else:
            a = int(np.argmax(Q[s]))

        r, t_next = step(t, a)

        if t_next < market.num_steps:
            s_next = state_to_index(battery.get_soc(), market.prices[t_next])
            td_target = r + gamma * np.max(Q[s_next])
        else:
            s_next = None
            td_target = r

        Q[s, a] += alpha * (td_target - Q[s, a])
        s = s_next if s_next is not None else s
        t = t_next

print("Learned Q-table (shape):", Q.shape)

# Test the agent
battery.reset(soc=initial_soc)
total_profit = 0.0
for t in range(market.num_steps):
    s = state_to_index(battery.get_soc(), market.prices[t])
    a = int(np.argmax(Q[s]))
    r, _ = step(t, a)
    total_profit += r
print(f"One-day greedy simulation profit: €{total_profit:.2f}, final {battery}")


plot_actions(battery=battery, market=market, timestep=timestep, initial_soc=initial_soc)














# # todo: plot history and compare with market prices
# import matplotlib.pyplot as plt

# history = battery.get_history()


# # Reconstruct SOC timeline from battery.history
# # battery.history entries are (soc_before, duration_min, action) where action: 1=charge,0=discharge
# soc_history = []
# time_steps = []
# actions = []
# dt = timestep  # minutes per timestep
# current_time = 0
# current_soc = initial_soc
# for entry in history:
#     soc_before, duration_min, action = entry
#     # duration may be dt (15) per step; number of timesteps this entry represents
#     steps = int(duration_min / dt) if duration_min >= dt else 1
#     for s in range(steps):
#         time_steps.append(current_time)
#         soc_history.append(current_soc)
#         actions.append(action)
#         # update soc according to action
#         if action == 1:
#             # charge
#             added = battery.power * (dt / 60)
#             current_soc = min(battery.capacity, current_soc + added)
#         else:
#             # discharge
#             removed = battery.power * (dt / 60)
#             current_soc = max(0, current_soc - removed)
#         current_time += 1

# # If history is empty, fallback to a single point
# if len(time_steps) == 0:
#     time_steps = [0]
#     soc_history = [battery.get_soc()]
#     actions = [None]

# # Convert market prices to align with time_steps (they are already 0..num_steps-1)
# price_ts = market.prices[:len(time_steps)] if len(market.prices) >= len(time_steps) else np.pad(market.prices, (0, len(time_steps)-len(market.prices)), constant_values=(market.prices[-1] if len(market.prices)>0 else 0))

# # Plot
# fig, ax1 = plt.subplots(figsize=(14,6), dpi=150)
# ax2 = ax1.twinx()

# # Price as stepped line
# ax1.step(range(len(price_ts)), price_ts, where='post', color='tab:blue', linewidth=2, label='Market price [€/kWh]')
# ax1.set_ylabel('Price [€/kWh]', color='tab:blue')
# ax1.tick_params(axis='y', labelcolor='tab:blue')

# # SOC as line
# ax2.plot(range(len(soc_history)), soc_history, color='tab:orange', marker='o', label='Battery SOC (kWh)')
# ax2.set_ylabel('SOC (kWh)', color='tab:orange')
# ax2.tick_params(axis='y', labelcolor='tab:orange')

# # Mark actions: green up for charge, red down for discharge
# for idx, act in enumerate(actions):
#     if act is None:
#         continue
#     if act == 1:
#         ax2.scatter(idx, soc_history[idx], color='green', marker='^', s=40, label='Charge' if idx==0 else "")
#     else:
#         ax2.scatter(idx, soc_history[idx], color='red', marker='v', s=40, label='Discharge' if idx==0 else "")

# ax1.set_title('Market prices and Battery actions over time')
# ax1.set_xlabel('Timestep (15 min)')
# ax1.grid(True, linestyle='--', alpha=0.4)
# fig.tight_layout()
# plt.legend(loc='upper left')
# plt.savefig('market_with_actions.png')
# print('Saved market_with_actions.png')

