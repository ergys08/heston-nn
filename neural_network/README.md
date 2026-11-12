# Neural network experiments

The state-to-angle SuperEncoder models, training functions, fidelity loss, and synthetic-state generators live here.

- `mlp.py`: SuperEncoder MLP.
- `loss.py`: fidelity objective.
- `train.py`: basic train_superencoder entry point.
- `training_workflow.py`: checkpointing, historical state construction, training, and evaluation.
- `synthetic_states.py`: smooth amplitude-space augmentation.
- `synthetic_random_states.py`: random-state control augmentation.

From the **repository root**, existing experiment commands still work:

```bash
python main.py
python run_random_control.py
python -m pytest tests/test_superencoder.py
```

Import functions as modules under `neural_network`:

```python
from neural_network.training_workflow import read_heston_states, load_checkpoint, evaluate_network
from neural_network.synthetic_states import generate_synthetic_states
```

Saved result directories and checkpoint paths remain unchanged.
