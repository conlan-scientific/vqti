
+ Use tab indentation in Python files.
+ Maximum line length of 80. Set a ruler in your IDE if necessary.

+ All function names should be verbs.
+ Class definitions are title case e.g. `MyClassName`, while function names are snake case e.g. `my_function_name`. Variable names are also snake case e.g. `my_variable_name`. 

+ A leading underscore e.g. `_some_function` can denote a *private* method, meaning it won't be imported by other modules are not intended to be accessed by users interfacing with a class or module.

+ Input arguments to functions should not be modified within the function call. This is especially important when dealing with data frames and objects in general.

+ Always establish a path to external files relative to the script or module that is importing them, e.g. `os.path.join(os.path.dirname(__file__), '..', 'my_data.csv')`

+ Assume your end user's working directory is the `src` folder. This simulates the effect of using `vqti` as a `pip` package, which it may become eventually.

+ Python filenames should all be snakecase and should not contain numbers, except for standalone scripts. Standalone scripts are not importable by other processes, and should begin with a number.

+ All importable code, should be in the `src/vqti` directory. As a consequence, standalone scripts will typically be in the `src` directory.

+ We're programming clear and clean code, where the code itself serves as documentation. Think about the end-user's experience of exploring the source code when organizing the repo.

+ Use `assert` statements and `if __name__ == '__main__':` clauses to test code within the same file the code is written.

+ Regarding the above, no files should have the word `test` in them, because that implies adherence to a specific testing framework. It is also inappropriate for users to import files with the word `test` in them.

+ Keep it DRY (don't repeat yourself). There should be a single source of truth for each distinct functionality in the repo. If there are multiple versions of a function, which we expect due to our commitment to code profiling work, it should clear which one is optimal and which one is recommended for production use.

+ Remember YAGNI (you aren't gonna need it). If there is an opportunity to right more stable, concise, and explicit code by limiting its functionality, then do it. Avoid accomodating use cases that are not presently unneeded. Never accomodate use cases that are untestatble. 

-----------

+ Revile *squishy* data structures and embrace *hard* data structures. Revile *permissive* functions and prefer *strict* functions. If an assumption about the input data is violated, we want to know about it, and we want to know exactly why. Be a defensive programmer.


```python
df1.merge(df2) # Squishy 
pd.concat([df1, df2], axis=1) # Hard

def calculate_sma(df):
	return pd.Series() # Squishy 
	return pd.Series([], index=df.index, dtype='float64', name='sma') # Hard

s.map(some_dict) # Squishy 
s.apply(lambda x: some_dict[x]) # Hard


# Non-unique indexes and column names are completely banned
series.loc['2020-01-01'] # returns a number for unique index
series.loc['2020-01-01'] # returns a series for non-unique index

df.loc['2020-01-01'] # returns a series for unique index
df.loc['2020-01-01'] # returns a data frame for non-unique index

```

