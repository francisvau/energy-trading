
class Battery:    
    def __init__(self, soc=15, capacity=30, power=5):
        if not (0 <= soc <= capacity):
            raise ValueError("SOC must be between 0 and capacity.")
        self.soc = soc # kWh
        self.capacity = capacity # kWh
        self.power = power # kW
        self.history = [] # list of (soc, time, action)

    def get_soc(self):
        return self.soc
    
    def charge(self, duration=15):
        """Charge the battery (duration in minutes)."""
        self.history.append((self.soc, duration, 1))
        energy_added = self.power * (duration / 60)
        soc_before = self.soc
        self.soc = min(self.capacity, self.soc + energy_added)
        return self.soc - soc_before
    
    def discharge(self, duration=15):
        """Discharge the battery (duration in minutes)."""
        self.history.append((self.soc, duration, 0))
        energy_removed = self.power * (duration / 60)
        soc_before = self.soc
        self.soc = max(0, self.soc - energy_removed)
        return self.soc - soc_before
    
    def get_history(self):
        return self.history

    def reset(self, soc=15):
        if not (0 <= soc <= self.capacity):
            raise ValueError("SOC must be between 0 and capacity.")
        self.history = []
        self.soc = soc

    def __repr__(self):
        return f"Battery(SOC={self.soc:.2f}/{self.capacity} kWh)"
