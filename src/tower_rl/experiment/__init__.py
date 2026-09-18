"""What a training run is measured, recorded and read by.

Run identity, the tracking port and its MLflow adapter, the aggregation of a
run's measurements into metrics and reports, and the statistics a comparison
between runs is read with.  Everything here is about *observing* training; it
depends on the environment and the learner and nothing depends on it.
"""
