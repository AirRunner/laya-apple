"""Structured errors. Every failure the runtime can detect has its own type.

The runtime never falls back silently: when a requested backend, device, artifact or
shape is not validated, one of these is raised instead.
"""


class LayaAppleError(Exception):
    """Base class for every laya-apple error."""


class UnsupportedModelError(LayaAppleError):
    """The model id is not one of the registered, pinned checkpoints."""


class InvalidRequestError(LayaAppleError, ValueError):
    """The questions or state do not follow the Laya question schema."""


class UnsupportedShapeError(LayaAppleError):
    """The request does not fit any validated shape for the requested device."""


class BackendUnavailableError(LayaAppleError):
    """A required runtime (MLX, coremltools) is not installed or cannot run here."""


class ArtifactError(LayaAppleError):
    """Base class for Core ML artifact problems."""


class ArtifactMissingError(ArtifactError):
    """No artifact exists in the cache for this model/revision/bucket."""


class ArtifactRevisionError(ArtifactError):
    """The artifact was built from a different model revision or different weights."""


class ArtifactIntegrityError(ArtifactError):
    """The artifact's files do not match the hashes recorded in its manifest."""


class ArtifactParityError(ArtifactError):
    """The artifact has not passed the parity gate, or its recorded result is a failure."""


class ComputeUnitMismatchError(ArtifactError):
    """The artifact would run on compute units other than the validated ones.

    Raised when a caller asks for different compute units, or when Core ML's compute plan
    for the loaded artifact is not the validated placement (for example an ANE artifact
    whose plan is not 100% Neural Engine because ANE compilation failed).
    """
