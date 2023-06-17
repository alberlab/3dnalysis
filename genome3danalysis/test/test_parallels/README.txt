May 2023

We checked that the radial profile avg and std matches
with the code I wrote for previous structural features computation.

We have to add the gap file in the pipeline.

Then we have to write the rest of the structural features.


June 9

We added the GAP file in the pipeline. We haven't tested it yet.

We also added the Lamina computation (we used a mathematical formula
that is different than how Asli did it). We will have to check that this method
matches the results of Asli's method.

Next time we have to write the Nuclear Body structural features,
then the TransAB, ICP and RG.


June 16

We added the LaminaTSA code and the Nuclear Body (for now speckle and nucleoli) code.

Next time we have to write the TSA part for the Nuclear Body. Instead of creating another python
file, we can just add in the body.py file a keyword that will tell the code to compute the TSA.

We can do it also for the Lamina, but we would have to change the way we do lamina.py (we just use
lamina_tsa.py). We could even incorporate in the same code radial, lamina and lamina_tsa.

Then we can finish the code by adding the TransAB, ICP and RG.

NOTE: I think that, to make the folder more clean, we can rename the codes with a _ at the beginning
(since they are hidden to the user). For example, lamina.py -> _lamina.py.
