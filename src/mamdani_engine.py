import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class GenericMamdaniEngine:

  def __init__(self):
    self.antecedents = {} #intrari
    self.consequents = {} #iesiri
    self.simulation = None #motorul care ruleaza calculele efectiv

  def add_antecedent(self, name, universe_range, terms_dict):
    """Adaugă o variabilă de intrare (Input) cu funcțiile sale de apartenență.

    Exemplu terms_dict: {'mic': [0, 0, 5], 'mare': [5, 10, 10]}
    """
    var = ctrl.Antecedent(universe_range, name)
    for term_name, params in terms_dict.items():
      if len(params) == 3:
        var[term_name] = fuzz.trimf(var.universe, params)  # Triunghiulară
      elif len(params) == 4:
        var[term_name] = fuzz.trapmf(var.universe, params)  # Trapezoidală
    self.antecedents[name] = var

  def add_consequent(self, name, universe_range, terms_dict):
    """Adaugă variabila de ieșire (Output / Regresie) cu termenii lingvistici."""
    var = ctrl.Consequent(universe_range, name)
    for term_name, params in terms_dict.items():
      if len(params) == 3:
        var[term_name] = fuzz.trimf(var.universe, params)
      elif len(params) == 4:
        var[term_name] = fuzz.trapmf(var.universe, params)
    self.consequents[name] = var

  def build_system(self, rules_list):
    """Construiește sistemul de control pe baza listei de reguli transmise."""
    system_ctrl = ctrl.ControlSystem(rules_list)
    self.simulation = ctrl.ControlSystemSimulation(system_ctrl)

  def predict_row(self, row_data):
    """Rulează inferența pentru un singur rând (dicționar {nume_input: valoare})."""
    for key, val in row_data.items():
      if key in self.antecedents:
        # Asigurăm că valoarea nu depășește universul de discurs definit
        u_min = self.antecedents[key].universe[0]
        u_max = self.antecedents[key].universe[-1]
        self.simulation.input[key] = np.clip(val, u_min, u_max)

    try:
      self.simulation.compute()
      # Returnează prima ieșire (consecventul)
      consequent_name = list(self.consequents.keys())[0]
      return self.simulation.output[consequent_name]
    except Exception:
      # În caz că o regulă nu se activează corect, returnează o valoare de fallback (de ex. media)
      return np.nan

  def predict_dataset(self, df, input_columns):
    """Aplică motorul fuzzy pe un întreg DataFrame pentru coloanele specificate."""
    predictions = []
    for _, row in df.iterrows():
      row_dict = {col: row[col] for col in input_columns}
      pred = self.predict_row(row_dict)
      predictions.append(pred)
    return np.array(predictions)