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


July 5

We added the TSA computation for the Nuclear Body.
The code has been incorporated in body.py, although it is used not efficiently in the main code
(we run it two times, could be done once instead).

We also added the TransAB computation. We tested that it works and produces a reasonable average track,
but we don't know if it agrees with Asli's previous code.

Next time we can just finish with ICP and RG.


July 7

We added the ICP and the RG computation. We haven't tested them, not even if they work, because Hoffman
was down (we worked on the local machine).

We should:
    1) Check that the code for ICP and RG works;
    2) Check that all the structural features we computed match with the previous code;
    3) Include the possibility of multiple RG computations;
    4) Include the chromosome surface localization feature;
    5) Merge the codes that are similar (radial, lamin and lamin_tsa, icp and transAB);
    6) Add a main file, like igm-run to run the whole pipeline;

Then (or before) we have to work on the Markov Clustering algorithm.


July 19

We checked that the ICP and RG codes work, at least computationally (there was a little bug in RG).

Now we're applying the pipeline to the H1 dataset (HDS). We are in the process of processing
the Markov clustering into our new pickle format. We will also need the GAP BedFile and the AB BedFile.


January 29, 2024

We finished setting up the data for the H1 run, and it's now running (it takes a lot of time to do trasAB).
We fixed a bug related to how pandas reads the csv files (we now use '\s+' as separator instead of '\t').

Next time - if the run on H1 finishes correctly - we should compare the results with the previous structural features.

If everything is okay we can publish this version with structural features and work on the Markov Clustering.

(points 3 4 5 6 of the previous list can be done later)


February 20, 2024

The run on H1 didn't finish: it got stuck on the transAB computation.
This is probably because the memory on the nodes is too small to handle the computation.

I have implemented a memory_efficient version of the transAB computation, and it seems to work.

However, I got a weird pickle error, and I decided to shift to HDF5 format for the data (something that I wanted to do anyway).


February 23, 2024

I changed the data structure from pickle to HDF5.

I checked that it all works fine with the test data.

When I tried it on the H1 data, now for some reason I get stuck at the first computation (radial).
I don't know what the issue is: it seems like it can finish the parallel function successfully, but it doesn't enter the reduce function.

I will have to understand what the issue is.
