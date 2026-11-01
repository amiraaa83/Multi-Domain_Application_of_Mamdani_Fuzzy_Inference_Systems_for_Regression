import numpy as np
import pandas as pd


def clean_missing_values(df, subset_columns=None):
  """Elimină rândurile cu valori nule doar pentru coloanele de interes."""
  if subset_columns:
    return df.dropna(subset=subset_columns).copy()
  return df.dropna().copy()


def scale_to_range(series, original_min, original_max, target_min, target_max):
  """Normalizează o serie numerică la un interval țintă (opțional,

  dacă vrei să aduci intrările direct în universul de discurs fuzzy).
  """
  scaled = (series - original_min) / (original_max - original_min)
  return scaled * (target_max - target_min) + target_min