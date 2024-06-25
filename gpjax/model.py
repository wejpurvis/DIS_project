"""
This module defines the GPJax implementation of the SIMM latent force model. As GPJax does not inherently support the sharing of parameters across mean functions and kernels, a custom model is defined (instead of using a `gpx.gps.ConjugatePosterior`)
"""
