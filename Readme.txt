README
======
Multi-Objective Deep Learning for Concurrent Cost-Schedule-Risk Prediction
in Construction Projects

Dublin Business School, MSc in Data Analytics
[Your Name] | Student Number: [XXXXXXXX]


WHAT'S IN THIS ARTEFACT
------------------------
Hi, and thanks for taking the time to look through this. This folder
contains everything needed to reproduce the results reported in Chapters
3 and 4 of my dissertation. Here's what each file is and why it's there:

1. pipeline.py
   This is the full pipeline in a single Python file. It does everything:
   turns the raw government contract data into a clean project-level
   dataset, engineers the cost/schedule/risk targets, trains the
   multi-task neural network, trains the Random Forest and single-task
   MLP baselines, and prints out the final comparison table. I kept it
   as one file on purpose so it's easy to read top to bottom and easy
   to re-run without hunting through several scripts.

2. projects_clean.csv
   This is the cleaned dataset after all the data engineering steps
   described in Section 3.3 (linking transactions into projects,
   removing right-censored recent-year projects, etc.). If you just
   want to check the model results without waiting for the raw data to
   be re-processed, this is the file to use.

3. raw_data_source.txt
   The raw dataset (FPDSData.csv) is about 256MB, which is too big to
   comfortably include here, so this file just has the direct download
   link and DOI instead. If you want to see the full raw-to-clean
   pipeline run from scratch, download it from there and follow the
   steps below.

4. figures/
   All the charts and terminal screenshots used in Chapters 3 and 4 -
   things like the data funnel diagram, the model architecture diagram,
   and the results comparison charts. These were generated directly
   from the code in this artefact, not made separately, so they should
   match exactly if you re-run everything.

5. results_log.txt
   The raw console output from the final run of the pipeline. If a
   number in my results table looks odd, this is where you can check
   exactly what the code printed.

6. This Readme.


HOW TO RUN IT
--------------
You'll need Python 3 with pandas, numpy, and scikit-learn installed.
Nothing exotic - I deliberately avoided needing PyTorch or TensorFlow
since I built the neural network directly in NumPy (more on that in
Section 3.5 of the dissertation, but the short version is: the sandbox
I developed this in didn't have deep learning libraries available, so
I wrote the forward/backward pass by hand instead, which turned out
fine for a model this size).

If you just want to see the results (recommended, this is quick):

    python pipeline.py --data projects_clean.csv --prepared

This trains everything and prints the comparison table straight to
your terminal in a minute or two.

If you want to rebuild projects_clean.csv from the raw data yourself
(this takes longer, maybe 5-10 minutes depending on your machine):

    1. Download FPDSData.csv using the link in raw_data_source.txt
    2. Run: python pipeline.py --raw FPDSData.csv --prepare-out projects_clean.csv
    3. Then run the command above as normal

If you're not sure whether your download has the same column names I
used, there's a quick check built in:

    python pipeline.py --inspect FPDSData.csv

This just prints out every column header in the file so you can compare
it against what the code expects (see config section near the top of
pipeline.py).


A COUPLE OF THINGS WORTH KNOWING
----------------------------------
- The results will not be bit-for-bit identical every time you run it.
  The neural network's starting weights and the order it sees training
  batches in are randomised, so you might see R-squared of 0.03 one run
  and 0.035 the next. This is normal and expected - the pattern in the
  results (multi-task network beating the single-task MLP, Random Forest
  still winning on the two classification tasks) holds up consistently
  even though the exact decimals move around a little.

- I did not have GPU access while building this, and honestly didn't
  need it - the dataset is tens of thousands of rows, not images or
  text, so a normal laptop CPU handles training in well under a minute
  for the neural network and a couple of minutes for the Random Forest
  baselines. If you're on a slower machine, the Random Forest step is
  the one that takes the longest, so that's the "check your coffee is
  still warm" step, not the neural network.

- If something in the code doesn't match a paragraph in my dissertation,
  trust the code - the dissertation was written to describe what the
  code actually does, but if I made a documentation slip anywhere,
  pipeline.py is the source of truth.

Thanks again for reading through this - happy to answer any questions
about how any part of it works.
