import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression(y_true, y_pred):
  """Calculează metricile de regresie și returnează un DataFrame (tabel)

  cu indicatorii cantitativi de eroare.
  """
  mae = mean_absolute_error(y_true, y_pred)
  mse = mean_squared_error(y_true, y_pred)
  rmse = np.sqrt(mse)
  r2 = r2_score(y_true, y_pred)

  metrics_dict = {
      'Metric': [
          'MAE (Mean Absolute Error)',
          'MSE (Mean Squared Error)',
          'RMSE (Root Mean Squared Error)',
          'R2 Score',
      ],
      'Value': [mae, mse, rmse, r2],
  }
  return pd.DataFrame(metrics_dict)


def plot_regression_results(y_true, y_pred, title='Real vs. Predicted'):
  """Generează scatter plot-ul comparativ între valorile reale și cele prezise."""
  plt.figure(figsize=(8, 6))
  sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, color='royalblue')
  min_val = min(min(y_true), min(y_pred))
  max_val = max(max(y_true), max(y_pred))
  plt.plot(
      [min_val, max_val],
      [min_val, max_val],
      'r--',
      lw=2,
      label='Ideal (y_true = y_pred)',
  )
  plt.xlabel('Ground Truth (Real Values)')
  plt.ylabel('Mamdani FIS Predictions')
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()


def plot_membership_functions(fis_engine):
  """Afișează graficele funcțiilor de apartenență pentru toate intrările

  și ieșiri din sistemul Mamdani.
  """
  print('--- Generare Grafice Funcții de Apartenență (Antecedente) ---')
  for name, var in fis_engine.antecedents.items():
    var.view()
    plt.title(f'Membership Functions - Input: {name}')
    plt.show()

  print('--- Generare Grafice Funcții de Apartenență (Consecvent) ---')
  for name, var in fis_engine.consequents.items():
    var.view()
    plt.title(f'Membership Functions - Output: {name}')
    plt.show()


def plot_control_surface(simulation, input1_name, input2_name, output_name):
  """Generează suprafața de control 3D (Control Surface Plot)

  pentru două variabile de intrare, menținând a treia constantă.
  """
  h_range = np.linspace(20, 150, 30)
  s_range = np.linspace(0, 21, 30)
  H, S = np.meshgrid(h_range, s_range)
  Z = np.zeros_like(H)

  wake_fixed = 1.0  # Valoare fixă pentru wake_ups (puține treziri)

  for i in range(H.shape[0]):
    for j in range(H.shape[1]):
      simulation.input[input1_name] = H[i, j]
      simulation.input[input2_name] = S[i, j]
      simulation.input['wake_ups'] = wake_fixed
      try:
        simulation.compute()
        Z[i, j] = simulation.output[output_name]
      except Exception:
        Z[i, j] = np.nan

  fig = plt.figure(figsize=(10, 7))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(H, S, Z, cmap='viridis', edgecolor='none', alpha=0.8)
  ax.set_xlabel('HRV')
  ax.set_ylabel('Day Strain')
  ax.set_zlabel('Sleep Performance')
  ax.set_title(
      '3D Control Surface Plot (HRV vs. Day Strain -> Sleep Performance)'
  )
  fig.colorbar(surf, shrink=0.5, aspect=5)
  plt.show()