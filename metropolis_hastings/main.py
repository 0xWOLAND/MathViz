import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import random
import os

class AdPlacementOptimizer:
    def __init__(self, data_path: str, budget: float = 1_000_000_000):
        """Initialize the optimizer with county data and budget"""
        # Ensure correct file path
        self.data_path = os.path.join(os.path.dirname(__file__), data_path)
        self.data = pd.read_csv(self.data_path)
        self.budget = budget
        
        # Clean data first
        self.clean_data()
        
        self.counties = self.data['County'].unique()
        self.n_counties = len(self.counties)
        
        # Ad channels and their base costs (scaled for major campaign)
        self.channels = {
            'TV': {
                'base_cost': 2_000_000,  # $2M base cost for TV campaign per county
                'reach_multiplier': 0.4,  # Reaches 40% of target audience
                'description': 'Television advertising including local stations and cable'
            },
            'Digital': {
                'base_cost': 800_000,    # $800K base cost for digital campaign per county
                'reach_multiplier': 0.35, # Reaches 35% of target audience
                'description': 'Social media, display ads, search, and streaming platforms'
            },
            'Grassroots': {
                'base_cost': 400_000,    # $400K base cost for grassroots campaign per county
                'reach_multiplier': 0.25, # Reaches 25% of target audience
                'description': 'Community events, local partnerships, and direct outreach'
            }
        }
        
        # Normalize demographic factors for scoring
        self.normalize_data()

    def clean_data(self):
        """Clean the data before normalization"""
        # Define numeric columns to clean
        numeric_columns = [
            'Education_Less_Than_9th_Percent',
            'Language_Isolation_Percent',
            'Age_18_39_Percent',
            'Age_18_39_People'
        ]
        
        for col in numeric_columns:
            if col in self.data.columns:
                # Replace 'data not available' with NaN
                self.data[col] = pd.to_numeric(
                    self.data[col].replace({'data not available': np.nan, 'N/A': np.nan}), 
                    errors='coerce'
                )
                
                # Fill NaN values with column mean only for numeric columns
                mean_value = self.data[col].mean()
                self.data[col] = self.data[col].fillna(mean_value)

    def normalize_data(self):
        """Normalize the demographic data for scoring"""
        cols_to_normalize = [
            'Education_Less_Than_9th_Percent',
            'Language_Isolation_Percent',
            'Age_18_39_Percent'
        ]
        
        for col in cols_to_normalize:
            if col in self.data.columns:
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                if max_val > min_val:  # Avoid division by zero
                    self.data[f'{col}_normalized'] = (self.data[col] - min_val) / (max_val - min_val)
                else:
                    self.data[f'{col}_normalized'] = 1.0

    def calculate_impact_score(self, allocation: Dict[str, Dict[str, float]]) -> float:
        """Calculate the impact score for a given allocation"""
        total_score = 0
        total_cost = 0
        
        for county, channels in allocation.items():
            county_data = self.data[self.data['County'] == county].iloc[0]
            
            # Population-weighted demographic factors
            population = float(county_data.get('Age_18_39_People', 0))  # Ensure float
            education_factor = float(county_data.get('Education_Less_Than_9th_Percent_normalized', 0))
            language_factor = float(county_data.get('Language_Isolation_Percent_normalized', 0))
            age_factor = float(county_data.get('Age_18_39_Percent_normalized', 0))
            
            county_score = 0
            county_cost = 0
            
            for channel, amount in channels.items():
                channel_info = self.channels[channel]
                channel_cost = amount * channel_info['base_cost']
                channel_reach = amount * channel_info['reach_multiplier'] * population
                
                # Calculate channel-specific impact with population weighting
                if channel == 'TV':
                    impact = channel_reach * (0.4 * age_factor + 0.3 * education_factor + 0.3 * language_factor)
                elif channel == 'Digital':
                    impact = channel_reach * (0.6 * age_factor + 0.2 * education_factor + 0.2 * language_factor)
                else:  # Grassroots
                    impact = channel_reach * (0.3 * age_factor + 0.4 * education_factor + 0.3 * language_factor)
                
                county_score += impact
                county_cost += channel_cost
            
            total_score += county_score
            total_cost += county_cost
        
        # Penalize if over budget
        if total_cost > self.budget:
            return -float('inf')
        
        # Add efficiency bonus if using budget well (within 5% of total)
        if total_cost > self.budget * 0.95:
            total_score *= 1.1
        
        return total_score

    def propose_new_allocation(self, current: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Generate a new proposed allocation"""
        new_allocation = {county: {channel: amount for channel, amount in channels.items()}
                         for county, channels in current.items()}
        
        # Randomly select 1-3 counties to modify
        n_counties = random.randint(1, 3)
        selected_counties = random.sample(list(new_allocation.keys()), n_counties)
        
        for county in selected_counties:
            # Randomly select a channel
            channel = random.choice(list(self.channels.keys()))
            
            # Modify allocation with smaller steps for more precise optimization
            delta = random.uniform(-0.2, 0.2)
            new_allocation[county][channel] = max(0, min(2, new_allocation[county][channel] + delta))
        
        return new_allocation

    def metropolis_hastings(self, n_iterations: int = 20000, temperature: float = 1.0) -> Tuple[Dict[str, Dict[str, float]], float]:
        """Run Metropolis-Hastings algorithm to optimize ad placement"""
        # Initialize with small random allocations
        current_allocation = {
            county: {channel: random.uniform(0, 0.5) for channel in self.channels.keys()}
            for county in self.counties
        }
        
        current_score = self.calculate_impact_score(current_allocation)
        best_allocation = current_allocation.copy()
        best_score = current_score
        
        # Track progress
        progress_interval = n_iterations // 10
        
        for i in range(n_iterations):
            proposed_allocation = self.propose_new_allocation(current_allocation)
            proposed_score = self.calculate_impact_score(proposed_allocation)
            
            # Calculate acceptance probability
            if proposed_score > current_score:
                acceptance_prob = 1.0
            else:
                acceptance_prob = np.exp((proposed_score - current_score) / temperature)
            
            if random.random() < acceptance_prob:
                current_allocation = proposed_allocation
                current_score = proposed_score
                
                if current_score > best_score:
                    best_allocation = current_allocation.copy()
                    best_score = current_score
            
            # Decrease temperature
            temperature *= 0.9999
            
            # Print progress
            if (i + 1) % progress_interval == 0:
                print(f"Progress: {(i + 1) // progress_interval * 10}% complete")
        
        return best_allocation, best_score

def main():
    try:
        # Use relative path to the CSV file
        data_path = 'hdpulse_data.csv'
        
        # Initialize optimizer with $1 billion budget
        optimizer = AdPlacementOptimizer(data_path)
        
        print("Starting optimization with $1 billion budget...")
        print("This will take a few minutes to complete.")
        
        best_allocation, best_score = optimizer.metropolis_hastings()
        
        # Print results
        print(f"\nOptimization completed with score: {best_score:,.2f}")
        print(f"Total budget: ${optimizer.budget:,.2f}")
        
        # Calculate and display total spend
        total_spend = 0
        county_totals = {}
        
        for county, channels in best_allocation.items():
            county_total = sum(amount * optimizer.channels[channel]['base_cost'] 
                             for channel, amount in channels.items())
            county_totals[county] = county_total
            total_spend += county_total
        
        print(f"Total allocated: ${total_spend:,.2f}")
        print(f"Budget utilization: {(total_spend/optimizer.budget)*100:.1f}%")
        
        # Print top 15 counties by allocation
        print("\nTop 15 counties by allocation:")
        for county, total in sorted(county_totals.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"\n{county}: ${total:,.2f}")
            county_data = optimizer.data[optimizer.data['County'] == county].iloc[0]
            print(f"Population (18-39): {int(county_data['Age_18_39_People']):,}")
            for channel, amount in best_allocation[county].items():
                channel_spend = amount * optimizer.channels[channel]['base_cost']
                if channel_spend > 0:
                    print(f"  {channel}: ${channel_spend:,.2f} "
                          f"({amount:.2f} units, {(channel_spend/total)*100:.1f}% of county budget)")
    
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
