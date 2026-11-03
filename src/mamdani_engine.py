import itertools

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class GenericMamdaniEngine:

  def __init__(self):
    self.antecedents = {} #intrari
    self.consequents = {} #iesiri
    self.simulation = None #motorul care ruleaza calculele efectiv
    self.term_params = {} #parametrii MF pentru fiecare variabila: {var: {term: params}}
    self.fallback_count = 0 #cate predictii au folosit valoarea de rezerva

  @staticmethod
  def _membership(universe, params):
    if len(params) == 3:
      return fuzz.trimf(universe, params)  # Triunghiulară
    if len(params) == 4:
      return fuzz.trapmf(universe, params)  # Trapezoidală
    raise ValueError(f'Unsupported MF params (need 3 or 4 values): {params}')

  def add_antecedent(self, name, universe_range, terms_dict):
    """Adaugă o variabilă de intrare (Input) cu funcțiile sale de apartenență.

    Exemplu terms_dict: {'mic': [0, 0, 5], 'mare': [5, 10, 10]}
    """
    var = ctrl.Antecedent(universe_range, name)
    for term_name, params in terms_dict.items():
      var[term_name] = self._membership(var.universe, params)
    self.antecedents[name] = var
    self.term_params[name] = dict(terms_dict)

  def add_consequent(self, name, universe_range, terms_dict,
                     defuzzify_method='centroid'):
    """Adaugă variabila de ieșire (Output / Regresie) cu termenii lingvistici.

    defuzzify_method: 'centroid', 'bisector', 'mom', 'som' sau 'lom'.
    """
    var = ctrl.Consequent(universe_range, name, defuzzify_method=defuzzify_method)
    for term_name, params in terms_dict.items():
      var[term_name] = self._membership(var.universe, params)
    self.consequents[name] = var
    self.term_params[name] = dict(terms_dict)

  def set_defuzzify_method(self, method):
    """Schimbă metoda de defuzzificare pentru toți consecvenții."""
    for var in self.consequents.values():
      var.defuzzify_method = method

  def term_centers(self, var_name):
    """Centrul (centroidul) fiecărui termen lingvistic al unei variabile."""
    if var_name in self.antecedents:
      var = self.antecedents[var_name]
    else:
      var = self.consequents[var_name]
    centers = {}
    for term_name, term in var.terms.items():
      centers[term_name] = float(fuzz.defuzz(var.universe, term.mf, 'centroid'))
    return centers

  def build_system(self, rules_list):
    """Construiește sistemul de control pe baza listei de reguli transmise."""
    system_ctrl = ctrl.ControlSystem(rules_list)
    self.simulation = ctrl.ControlSystemSimulation(system_ctrl)
    self.fallback_count = 0

  def _fallback_value(self):
    """Valoarea de rezervă: centrul universului de discurs al ieșirii."""
    universe = list(self.consequents.values())[0].universe
    return float((universe[0] + universe[-1]) / 2.0)

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
      # Nicio regulă nu s-a activat: returnăm centrul universului de ieșire
      self.fallback_count += 1
      return self._fallback_value()

  def predict_dataset(self, df, input_columns):
    """Aplică motorul fuzzy pe un întreg DataFrame pentru coloanele specificate."""
    predictions = []
    for _, row in df.iterrows():
      row_dict = {col: row[col] for col in input_columns}
      pred = self.predict_row(row_dict)
      predictions.append(pred)
    return np.array(predictions)

  def firing_strengths(self, df, input_columns):
    """Calculează gradul de activare (AND = min) pentru fiecare combinație
    de termeni ai intrărilor, pe fiecare rând din df.

    Returnează (combos, W) unde W are forma (n_rows, n_combos).
    """
    memberships = []
    term_names = []
    for col in input_columns:
      var = self.antecedents[col]
      x = np.clip(df[col].to_numpy(dtype=float), var.universe[0], var.universe[-1])
      names = list(var.terms.keys())
      term_names.append(names)
      memberships.append({
          t: fuzz.interp_membership(var.universe, var[t].mf, x) for t in names
      })

    combos = list(itertools.product(*term_names))
    W = np.empty((len(df), len(combos)))
    for k, combo in enumerate(combos):
      w = np.ones(len(df))
      for i, term in enumerate(combo):
        w = np.minimum(w, memberships[i][term])
      W[:, k] = w
    return combos, W

  def learn_rule_consequents(self, df, input_columns, output_name,
                             min_support=1.0, default_rules=None):
    """Învață consecventul fiecărei reguli din date (stil Wang–Mendel).

    Pentru fiecare combinație de termeni ai intrărilor se calculează media
    țintei ponderată cu gradul de activare al regulii, apoi se alege termenul
    de ieșire al cărui centru este cel mai apropiat de această medie.
    Regulile cu suport (suma activărilor) sub min_support păstrează
    consecventul din default_rules (dacă există) sau sunt omise.

    Returnează un DataFrame cu câte un rând pentru fiecare regulă.
    """
    combos, W = self.firing_strengths(df, input_columns)
    y = df[output_name].to_numpy(dtype=float)
    centers = self.term_centers(output_name)
    term_list = list(centers.keys())
    center_vals = np.array([centers[t] for t in term_list])

    rows = []
    for k, combo in enumerate(combos):
      support = float(W[:, k].sum())
      if support >= min_support:
        target_mean = float((W[:, k] * y).sum() / support)
        term = term_list[int(np.argmin(np.abs(center_vals - target_mean)))]
        source = 'data'
      else:
        target_mean = np.nan
        term = default_rules.get(combo) if default_rules else None
        source = 'expert' if term else 'none'
      row = dict(zip(input_columns, combo))
      row.update({
          'support': support,
          'weighted_mean': target_mean,
          output_name: term,
          'source': source,
      })
      rows.append(row)
    return pd.DataFrame(rows)

  def rules_from_table(self, rule_table, input_columns, output_name):
    """Construiește obiectele ctrl.Rule dintr-un tabel de reguli."""
    rules = []
    for _, r in rule_table.iterrows():
      if r[output_name] is None or pd.isna(r[output_name]):
        continue
      antecedent = None
      for col in input_columns:
        term = self.antecedents[col][r[col]]
        antecedent = term if antecedent is None else antecedent & term
      rules.append(ctrl.Rule(antecedent, self.consequents[output_name][r[output_name]]))
    return rules
