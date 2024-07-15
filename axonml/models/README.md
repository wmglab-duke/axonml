## Implementing your own surrogate fiber model

We implement the MRG axon model, however you may wish to simulate some other physics.

To implement your own axon type, subclass `axonml.models.Axon`. 

Parameters can be specified in the class attribute `params`, a dictionary consisting of subgroups, and parameter names and their values. For example, for the Hodgkin-Huxley model:

```python
from axonml.models import Axon

class HH(Axon):
    params = {
        'membrane' : {
            'cm': 1e-3,
            'rhoa': 70.0
        }
    }

```

All such declared parameters will be instantiated as instance attributes of your `Axon` subclass with type `torch.nn.Parameter`. By default, no parameters can be trained; to train a parameter, call `axon.unfreeze(parameter)`, e.g. `axon.unfreeze('cm')` to train the membrane capacitance.