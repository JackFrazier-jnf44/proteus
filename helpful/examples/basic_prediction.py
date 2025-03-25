from src.interfaces import ModelInterface, BaseModelConfig
from src.core.visualization import plot_structure

# Initialize model interface
config = BaseModelConfig(
    name="esm2_t33_650M_UR50D",
    model_type="esm",
    output_format="pdb"
)
model = ModelInterface(config)

# Predict structure
sequence = "MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF"
result = model.predict_structure(sequence)

# Visualize result
plot_structure(
    structure_file=result["structure"],
    confidence_scores=result["confidence"],
    output_file="structure_visualization.png"
)