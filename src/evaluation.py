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


def plot_control_surface(
    simulation,
    input1_name,
    input2_name,
    output_name,
    x_range,
    y_range,
    fixed_inputs=None,
    labels=None,
    title=None,
    resolution=30,
):
  """Generează suprafața de control 3D (Control Surface Plot) pentru două
  variabile de intrare, menținând celelalte intrări la valori fixe.

  x_range / y_range: (min, max) pentru cele două intrări variate.
  fixed_inputs: dicționar {nume_input: valoare} pentru restul intrărilor.
  labels: (eticheta_x, eticheta_y, eticheta_z); implicit numele variabilelor.
  """
  fixed_inputs = fixed_inputs or {}
  x_label, y_label, z_label = labels or (input1_name, input2_name, output_name)
  if title is None:
    fixed_txt = ', '.join(f'{k}={v}' for k, v in fixed_inputs.items())
    title = f'Control Surface: {x_label} vs. {y_label} -> {z_label}'
    if fixed_txt:
      title += f' ({fixed_txt})'

  x_vals = np.linspace(x_range[0], x_range[1], resolution)
  y_vals = np.linspace(y_range[0], y_range[1], resolution)
  X, Y = np.meshgrid(x_vals, y_vals)
  Z = np.zeros_like(X)

  for i in range(X.shape[0]):
    for j in range(X.shape[1]):
      simulation.input[input1_name] = X[i, j]
      simulation.input[input2_name] = Y[i, j]
      for name, value in fixed_inputs.items():
        simulation.input[name] = value
      try:
        simulation.compute()
        Z[i, j] = simulation.output[output_name]
      except Exception:
        Z[i, j] = np.nan

  fig = plt.figure(figsize=(10, 7))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.set_zlabel(z_label)
  ax.set_title(title)
  fig.colorbar(surf, shrink=0.5, aspect=5, label=z_label)
  plt.show()


def plot_data_distributions(df, columns=None, figsize=(12, 8)):
  """Generates a grid of histograms with KDE curves for specified numerical
  columns to analyze data distribution, skewness, and spread across any dataset.
  """
  if columns is None:
    columns = df.select_dtypes(include=[np.number]).columns.tolist()

  n_cols = len(columns)
  n_rows = (n_cols + 2) // 3
  fig, axes = plt.subplots(n_rows, min(n_cols, 3), figsize=figsize)
  axes = np.array(axes).flatten()

  for i, col in enumerate(columns):
    if i < len(axes):
      sns.histplot(df[col], kde=True, ax=axes[i], color='teal', bins=30)
      axes[i].set_title(f'Distribution: {col}')
      axes[i].grid(True, alpha=0.3)

  # Hide any unused subplots if grid is larger than column count
  for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

  plt.tight_layout()
  plt.show()


def plot_universe_percentiles(
    df, target_col, percentiles=[0.01, 0.25, 0.50, 0.75, 1.0]
):
  """Visualizes the distribution of a target/consequent variable alongside
  its key calculated statistical percentiles. Ideal for mapping out
  fuzzy linguistic boundaries (e.g., poor, average, excellent) without dead
  zones.
  """
  plt.figure(figsize=(9, 5))
  sns.histplot(df[target_col], kde=True, color='royalblue', bins=40)

  vals = [df[target_col].quantile(p) for p in percentiles]
  labels = [f'{int(p*100)}th ({v:.1f})' for p, v in zip(percentiles, vals)]
  colors = ['red', 'orange', 'green', 'purple', 'brown']

  for val, label, color in zip(vals, labels, colors):
    plt.axvline(val, color=color, linestyle='--', linewidth=2, label=label)

  plt.title(f'Percentile Calibration Map: {target_col}')
  plt.xlabel(target_col)
  plt.ylabel('Density / Count')
  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.grid(True, alpha=0.3)
  plt.tight_layout()
  plt.show()