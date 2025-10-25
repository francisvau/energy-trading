import matplotlib.pyplot as plt
import numpy as np

from battery import Battery
from market import Market

def plot_actions(battery, market, timestep, initial_soc):
    history = battery.get_history()
    soc_history = []
    time_steps = []
    actions = []
    dt = timestep
    current_time = 0
    current_soc = initial_soc
    for entry in history:
        soc_before, duration_min, action = entry
        steps = int(duration_min / dt) if duration_min >= dt else 1
        for s in range(steps):
            time_steps.append(current_time)
            soc_history.append(current_soc)
            actions.append(action)
            if action == 2:
                # charge
                added = battery.power * (dt / 60)
                current_soc = min(battery.capacity, current_soc + added)
            elif action == 0:
                # discharge
                removed = battery.power * (dt / 60)
                current_soc = max(0, current_soc - removed)
            # if action == 1 (idle): do nothing
            current_time += 1

    # Convert market prices to align with time_steps
    price_ts = market.prices[:len(time_steps)] if len(market.prices) >= len(time_steps) else np.pad(
        market.prices,
        (0, len(time_steps) - len(market.prices)),
        constant_values=(market.prices[-1] if len(market.prices) > 0 else 0)
    )

    # Plot
    fig, ax1 = plt.subplots(figsize=(14,6), dpi=150)
    ax2 = ax1.twinx()

    # Price as stepped line
    ax1.step(range(len(price_ts)), price_ts, where='post', color='tab:blue', linewidth=2, label='Market price [€/kWh]')
    ax1.set_ylabel('Price [€/kWh]', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # SOC as line
    ax2.plot(range(len(soc_history)), soc_history, color='tab:orange', label='Battery SOC (kWh)')
    ax2.set_ylabel('SOC (kWh)', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # Mark actions
    for idx, act in enumerate(actions):
        if act is None:
            continue
        if act == 2:
            ax2.scatter(idx, soc_history[idx], color='green', marker='^', s=40, label='Charge' if idx==0 else "")
        elif act == 0:
            ax2.scatter(idx, soc_history[idx], color='red', marker='v', s=40, label='Discharge' if idx==0 else "")
        elif act == 1:
            ax2.scatter(idx, soc_history[idx], color='gray', marker='o', s=30, label='Idle' if idx==0 else "")

    ax1.set_title('Market prices and Battery actions over time')
    ax1.set_xlabel('Timestep')
    ax1.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    plt.savefig('market_with_actions.png')
    print('Saved market_with_actions.png')
