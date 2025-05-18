Okay so this is everything I need to get done with before I meet Shaon tomorrow:\

1. Update yourself on the math behind a VAE.
2. Specifically, include some schematic detailing the exact model that David Kyle uses.
3. Next, go to the latent space. Shuffle the 50 years of data and generate as many possible randomly chosen time-series vectors as you can. Enough so that the elliptical trend is clear and visible when plotting the PCA.
4. Separate out your code into modules, and make sure to save the model weights.
5. Then, based on hypothesised variables (temperature, day of year, month of year, quarter of the year, hour of day), plot the data colourwise, and see if there are any trends you can see on the PCA plots
6. Try other architectures, in particular, vary the latent space dimension from 2 to say 10 or 20, and see what you find. Do the same analysis for each of these different dimensions.
7. Think about incorporating LSTMs into this model, and whether it would be feasible or not, and what sorts of patterns we can uncover and unravel from it.
8. Lastly, if we are going to use the same ideas for the actual pipeline, think about how to adapt this for multivariate time series, and also how we might have to change the architecture and what the inputs to the model will be. 