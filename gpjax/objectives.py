"""
This module defines a custom objective function for the SIMM latent force model. The objective function is a slight modification of `gpx.objectives.ConjugateMLL` to allow passing of a custom model (instead of using a `gpx.gps.ConjugatePosterior`). The objective function is used in the training of the model.
"""
