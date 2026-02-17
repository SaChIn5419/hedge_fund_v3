import json
import random

class StrategyGenome:
    def __init__(self):
        # Default DNA (Parameters)
        self.dna = {
            "lookback_energy": 20,
            "lookback_vp": 60,
            "lookback_gaussian": 140,
            "vol_spike_threshold": 0.02,
            "max_order_chunk": 0.25
        }
        self.generation = 1

    def get_gene(self, key):
        return self.dna.get(key)

    def mutate(self):
        """
        Slightly alters parameters to find better fit. 
        In production, this runs on historical replay, NOT live.
        """
        mutation_rate = 0.1
        
        if random.random() < mutation_rate:
            # Example: Mutate Lookback Energy
            change = random.choice([-1, 1])
            self.dna["lookback_energy"] += change
            print(f"MUTATION: lookback_energy -> {self.dna['lookback_energy']}")
            self.generation += 1

    def load_dna(self, filepath="data/genome.json"):
        try:
            with open(filepath, "r") as f:
                self.dna = json.load(f)
        except:
            pass # Use defaults

    def save_dna(self, filepath="data/genome.json"):
        with open(filepath, "w") as f:
            json.dump(self.dna, f, indent=4)
