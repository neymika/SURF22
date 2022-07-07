# SURF22
 Summer Research

## Note from Week 3!!
  * Please copy the absolute path to the SURF22 project location and run the following command in order to ensure all scripts work as intended:
    * ```python
      export PYTHONPATH="${PYTHONPATH}:path/to/SURF22"
      ```
    * For example, on my Mac I used the following command:
    * ```python
      export PYTHONPATH="${PYTHONPATH}:/Users/neymikajain/Desktop/SURF22/"
      ```

## :worried: Week 1
  * 6/14: Got COVID :sob: :mask: :anger:... ON THE VERY FIRST DAY! Just my luck...
  * Worked on basic implementations of least squares solution for basic estimator in jupyter notebook and python script.
    * Requires the following package installations:
      * ```python
        pip3 install numpy
        ```
    * To run file:
      * ```python
        python3 commonsense.py
        ```
  * Worked on basic implementation of mult-fidelity approach based on parameters and functions f^(1) and f^(3) found in section 4.1 of [Multifidelity Monte Carlo Estimation of Variance and Sensitivity Indices](https://www.dropbox.com/s/y77c42t9po52384/QPOVW_mfgsa_juq2018.pdf?dl=0) by E. QIAN, B. PEHERSTORFER, D. O'MALLEY, V. VESSELINOV, AND K. WILLCOX
    * Requires the following package installations
      * ```python
        pip3 install numpy
        pip3 install bokeh
        pip3 install selenium
        conda install -c conda-forge firefox geckodriver
        ```
    * (Week 2) Or can use the following package installations
      * ```python
        pip3 install numpy
        pip3 install matplotlib
        ```
    * Initial parameter values are based on paper. To run the script either call
      * ```python
        python3 toymultifidelity.py filename.png
        ```
      or
      * ```python
        python3 toymultifidelity.py filename.png a b lowsample highsample budget alpha
        ```

## :weary: Week 2
:disappointed: :shit: :skull:
  * 6/21: Still have COVID.

:satisfied: :sunglasses: :dizzy:
  * 6/22: NO LONGER HAVE COVID!!!
  * Bokeh is a problem child sometimes, so in case of any issues, can just run the same code the same way but make sure matplotlib is at least installed.
  * Made a quick script of the steepest descent code. ~~Will make clean and debug.~~
  * Debugged steepest descent code. Still takes much longer compared to mentor's version (140 iterations vs. 1017 iterations), but trend is linear. Starting points are different so that could be a possible reason.
  * Basic implementation of steepest descent with backtracking using a homework problem dataset. Have a few questions however.
    * ~~Where do we ever introduce ridge regression in the GD problem? In mentor's code it is only ever introduced in the least squares solution.~~
      * Unnecessary in the GD case apparently.
    * ~~Mentor and my original function and derivates are different. Should ask why.~~
      * They are not different! Yet they are producing different results, so must be debugged.
    * ~~Mentor's code always starts with initial alpha instead of updated alpha. Why?~~
      * Did not really answer, but I imagine its to ensure that the search direction condition is met everytime?
    * Requires the following package installations
      * ```python
        pip3 install numpy
        pip3 install pandas
        pip3 install bokeh
        pip3 install selenium
        conda install -c conda-forge firefox geckodriver
        ```
    * Or can use the following package installations
      * ```python
        pip3 install numpy
        pip3 install pandas
        pip3 install matplotlib
        ```
    * To run file:
      * ```python
        python3 steepestdescent.py
        ```
  * Basic implementation of steepest descent with backtracking using a toy dataset where y = .5*x + 2 + $\epsilon$
    * Requires the same package installations as above.
    * To run file:
      * ```python
        python3 toysteepestdescent.py
        ```
  * Basic implementation of SGD with minibatch 20 for homework problem dataset.
    * Requires the same package installations as above.
    * To run file:
      * ```python
        python3 sgd.py
        ```
  * Basic implementation of SGD with minibatch 20 for toy dataset where y = .5*x + 2 + $\epsilon$.
    * Requires the same package installations as above.
    * To run file:
      * ```python
        python3 toysgd.py
        ```
  * Fixed and cleaned up non-toy steepest descent and SGD files. Issue was lambda value.
  * Comparison of SGD and GD with varying minibatch sizes for toy dataset example where y = .5*x + 2 + $\epsilon$.
    * Requires the same package installations as above.
    * To run file:
      * ```python
        python3 toy.py
        ```
  ## :anger: :speak_no_evil: Week 3
  * Changed convergence criteria for SD back to grad J... Resulted in SD never converging for small tau of 1e-6. Changed back to delta theta criteria instead to match SGD.
  * :white_check_mark: Keep the same criteria for sgd. Further cleaned up the code such that a single data set could be used across the board. Made SGD parallel to reduce the time taken for code to run.
  * Made tau even smaller (1e-6!). Saw some interesting results that I want to ask Pan about the next time we meet.
  * Added SVRG implementation script. Is not really working as I think it should. Going to debug.
  * Comparison of SGD and GD with varying minibatch sizes for unknown homework dataset example.
    * Requires the same package installations as above. For parallel calls, requires the following package AND THE PATH ADDITION AT THE TOP OF THE README
      * ```python
        pip3 install ray
        ```
    * To run file with parallel calls:
      * ```python
        python3 unknown.py
        ```
    * To run file with sequential calls (Takes longer):
    * ```python
      python3 unknown_sequential.py
      ```
  ## :dancer: :worried: Week 4
  * Mostly just worked on the interim report.
  * To reproduce the main plots from the report and the multifidelity results use the same package installations as above.
    * To run gradient descent plotting files:
      * ```python
        cd Week3
        python3 unknown_paper.py
        python3 toysvrg.py
        ```
    * To run multifidelity file:
      * ```python
        cd Week1
        python3 toymultifidelity.py
        ```
