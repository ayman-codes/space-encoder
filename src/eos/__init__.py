"""Earth observation hybrid autoencoder engine.

Importing the package pins every temporary artifact to the repository scratch
cache so pytest basetemp folders and torch Inductor caches can never land in
the tracked tree, regardless of the invoking shell. The guarantee must live at
package entry because the torch Inductor cache directory is created during
import of the dataset and training modules, before any entry point runs.
"""

from __future__ import annotations

from eos.tempdir import use_repo_scratch_space

use_repo_scratch_space()

__version__ = "0.1.0"
