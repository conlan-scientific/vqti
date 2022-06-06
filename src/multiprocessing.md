


#### 1 Python multiprocessing

+ Low-level
+ Close to the metal
+ Fine-grained control
+ Inflexbile
+ Python built-in
+ Not recommended
+ Supports multi-threading better than parallelization

#### 2 Joblib

+ This one is use by scikit-learn (n_jobs parameter)
+ Uses something called cloudpickle, https://github.com/cloudpipe/cloudpickle
+ Makes you write bad code (only works with a subset of Python's features)
+ Recommended for very small easy jobs that work off of 3rd party code (i.e. I/O and scikit-learn stuff)

#### 3 Luigi

+ Uses UNIX "fork" to parallelize
+ Rock solid, but all I/O happens on disk
+ Airflow is the same thing, but more heavyweight
+ I recommend this for optimizations

#### 4 Opening multi terminal tabs

+ Just do it.

