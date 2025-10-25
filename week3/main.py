""" Start coding in this main file
"""

# increase action space
# play around with parameters (learning curve)
# optimization problem
# efficiency
# other: more price, perhaps real prices (belpex for day ahead prices)

import numpy as np
import random

from battery import Battery
from market import Market
from plotter import plot_actions


# ------- 1. Market and battery -------
timestep = 15
initial_soc = 5
capacity = 10
battery = Battery(soc=initial_soc, capacity=capacity, power=5)
market = Market(pattern="square", duration_hours=24, timestep_minutes=timestep)
market.plot()


# ------- 2. Q learning setup + hyperparameters -------
n_states = (capacity + 1) * 2
actions = [0, 1, 2]
Q = np.zeros((n_states, len(actions)))
alpha = 0.05
gamma = 0.95
epsilon = 0.2
n_episodes = 500

def step(time_index, action):
    price = market.prices[time_index]
    reward = 0.0
    if action == 0: # discharge
        delivered = -battery.discharge(duration=timestep)
        reward = price * delivered
    elif action == 1: # idle
        battery.idle(duration=timestep)
        reward = 0.0
    else: # charge
        stored = battery.charge(duration=timestep)
        reward = -price * stored
    next_index = time_index + 1
    return reward, next_index

def state_to_index(soc, price):
    soc_bin = int(round(soc))
    price_level = 1 if price > 0 else 0
    return soc_bin + price_level * (capacity + 1)


# ------- 3. Training -------
for _ in range(n_episodes):
    battery.reset(soc=initial_soc)
    t = 0
    s = state_to_index(battery.get_soc(), market.prices[t])
    while t < market.num_steps:
        if random.random() < epsilon:
            a = random.choice([0, 1, 2])
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

print("Q-table shape:", Q.shape)


# ------- 4. Test the agent -------
battery.reset(soc=initial_soc)
total_profit = 0.0
for t in range(market.num_steps):
    s = state_to_index(battery.get_soc(), market.prices[t])
    a = int(np.argmax(Q[s]))
    r, _ = step(t, a)
    total_profit += r
print(f"One-day simulation profit: €{total_profit:.2f}")

# Plot results
plot_actions(battery=battery, market=market, timestep=timestep, initial_soc=initial_soc)
    
