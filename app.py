import streamlit as st
import pandas as pd
import numpy as np

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from mri_engine import MRIEngine

st.title("Nathan's MRI Prediction Engine")

engine = MRIEngine()
engine.read_flair()

print("Success!")
