from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import os

os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

model = load_model("dog_cat_model.keras")

plot_model(model,
            to_file='architecture.png',
            show_shapes=True,
            show_layer_names=True,
            dpi=96,
            rankdir='TB')

print("architecture.png created")