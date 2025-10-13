import random
import numpy as np
import matplotlib.pyplot as plt

class Market:
    def __init__(self, duration_hours=24, timestep_minutes=15, pattern="square", high_price=10, low_price=-10):
        if not (low_price < high_price):
            raise ValueError("Lowest price must be lower than highest price.")
        self.duration_hours = duration_hours
        self.timestep_minutes = timestep_minutes
        self.pattern = pattern
        self.high_price = high_price
        self.low_price = low_price
        self.num_steps = int(duration_hours * 60 / timestep_minutes)
        self.prices = self._generate_prices()
    
    def _generate_prices(self):
        if self.pattern == "square":
            return self._generate_square_wave()
        elif self.pattern == "sinus":
            return self._generate_sinus_wave()
        else:
            raise ValueError(f"Unknown pattern type: {self.pattern}")
    
    def _generate_square_wave(self):
        prices = []
        toggle = random.choice([self.high_price, self.low_price])
        i = 0
        while i < self.num_steps:
            block_length = random.randint(2, 8) # how many timesteps this block lasts
            for _ in range(block_length):
                if i >= self.num_steps:
                    break
                prices.append(toggle)
                i += 1
            toggle = self.high_price if toggle == self.low_price else self.low_price
        return np.array(prices)
    
    def _generate_sinus_wave(self):
        x = np.linspace(0, 2 * np.pi * 4, self.num_steps, endpoint=False)
        mid = (self.high_price + self.low_price) / 2
        amplitude = (self.high_price - self.low_price) / 2
        prices = mid + amplitude * np.sin(x)
        return prices
    
    def plot(self):
        plt.figure(figsize=(16, 9), dpi=150)
        x = range(len(self.prices))
        plt.step(x, self.prices, where="post", linewidth=2)
        plt.title("Imbalance Market Prices")
        plt.xlabel("Timestep (15 min each)")
        plt.ylabel("Price [€/MWh]")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("market_prices.png")
        print("Saved plot to market_prices.png")
