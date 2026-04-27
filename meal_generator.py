import numpy as np
import pandas as pd

class MealGenerator:
    def __init__(self, food_df):
        """
        Initialize the MealGenerator with the food catalog.
        
        Args:
            food_df (pd.DataFrame): The cleaned food catalog containing Recipes and Macros.
        """
        self.food_df = food_df.copy()
        # Drop rows with missing crucial macro data to avoid math errors during simulation
        self.food_df = self.food_df.dropna(subset=['Calories', 'Protein', 'Carbs', 'Fat'])
        
    def generate_meal_plan(self, target_cals, target_protein, target_carbs, target_fat, num_meals=7, iterations=10000):
        """
        Use a Monte Carlo random search approach to find a combination of meals that 
        closely match the target macronutrients and are distributed properly across the day.
        
        Args:
            target_cals (float): Target daily Calories (Calculated as Pro*4 + Carb*4 + Fat*9)
            target_protein (float): Target daily Protein (g)
            target_carbs (float): Target daily Carbs (g)
            target_fat (float): Target daily Fat (g)
            num_meals (int): Number of recipes to pick (default 7: Breakfast x2, Lunch x3, Dinner x2)
            iterations (int): Number of random combinations to test
            
        Returns:
            tuple: (best_meal_plan_df, best_error_score, actual_totals_dict)
        """
        best_error = float('inf')
        best_indices = None
        
        # We now include a Balance weight to ensure calories are spread sensibly
        weights = {
            'Calories': 0.35,
            'Protein': 0.35,
            'Carbs': 0.1,
            'Fat': 0.1,
            'Balance': 0.1
        }
        
        # Pre-extract numpy arrays for fast computation during the loop
        cal_array = self.food_df['Calories'].values
        pro_array = self.food_df['Protein'].values
        carb_array = self.food_df['Carbs'].values
        fat_array = self.food_df['Fat'].values
        
        num_recipes = len(self.food_df)
        
        # Calculate expected calories for each meal based on the 30/40/30 distribution
        expected_breakfast_cals = target_cals * 0.30
        expected_lunch_cals = target_cals * 0.40
        expected_dinner_cals = target_cals * 0.30
        
        for _ in range(iterations):
            # Randomly pick 'num_meals' indices (7 by default)
            idx = np.random.choice(num_recipes, num_meals, replace=False)
            
            # Calculate sum of macros for this combination
            sum_cals = np.sum(cal_array[idx])
            sum_pro = np.sum(pro_array[idx])
            sum_carbs = np.sum(carb_array[idx])
            sum_fat = np.sum(fat_array[idx])
            
            # --- Macro Errors ---
            err_cals = abs(target_cals - sum_cals) / max(target_cals, 1)
            err_pro = abs(target_protein - sum_pro) / max(target_protein, 1)
            err_carbs = abs(target_carbs - sum_carbs) / max(target_carbs, 1)
            err_fat = abs(target_fat - sum_fat) / max(target_fat, 1)
            
            # --- Caloric Balance Error ---
            if num_meals == 7:
                breakfast_cals = np.sum(cal_array[idx[0:2]])
                lunch_cals = np.sum(cal_array[idx[2:5]])
                dinner_cals = np.sum(cal_array[idx[5:7]])
                
                # Measure how far each meal deviates from its expected caloric share
                err_balance = (
                    abs(breakfast_cals - expected_breakfast_cals) +
                    abs(lunch_cals - expected_lunch_cals) +
                    abs(dinner_cals - expected_dinner_cals)
                ) / max(target_cals, 1)
            else:
                err_balance = 0.0 # Fallback if someone changes num_meals manually
            
            # Apply user-defined weights
            total_error = (
                err_cals * weights['Calories'] +
                err_pro * weights['Protein'] +
                err_carbs * weights['Carbs'] +
                err_fat * weights['Fat'] +
                err_balance * weights['Balance']
            )
            
            # Keep track of the best combination
            if total_error < best_error:
                best_error = total_error
                best_indices = idx
                
        # Fetch the winning recipes from the dataframe
        best_plan_df = self.food_df.iloc[best_indices].copy()
        
        actual_totals = {
            'Calories': np.sum(cal_array[best_indices]),
            'Protein': np.sum(pro_array[best_indices]),
            'Carbs': np.sum(carb_array[best_indices]),
            'Fat': np.sum(fat_array[best_indices])
        }
        
        return best_plan_df, best_error, actual_totals
