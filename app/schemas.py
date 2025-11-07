from typing import Dict, List, Union

from pydantic import BaseModel


class PredictIn(BaseModel):
    """Prediction input schema for the API.

    Accepts either an ordered list of numeric feature values or a mapping
    from feature names to numeric values. The service will support both
    forms for convenience during integration and testing.
    """
    features: Union[List[float], Dict[str, float]]


class PredictOut(BaseModel):
    """Prediction output schema returned by the API.

    Includes the predicted positive class probability, the hard label
    derived from the current decision threshold, and the end-to-end
    service-side latency measurement in milliseconds.
    """
    proba: float
    label: int
    latency_ms: float


class MetricsOut(BaseModel):
    """Service metrics schema for latency and throughput.

    Reports the number of recorded requests in the buffer, tail latency
    percentiles (p50, p95, p99) expressed in milliseconds, and a rough
    requests-per-second estimate for dashboarding.
    """
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    rps: float


class HealthOut(BaseModel):
    """Service health and model metadata schema.

    Indicates whether the model artifact is loaded and exposes minimal
    model metadata such as version, trained timestamp, and calibration type.
    """
    status: str
    model_loaded: bool
    model_info: Dict[str, Union[str, int, float, bool]]
