# SURF22
 Summer Research

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
  * 6/21: Still have COVID. :disappointed: :shit: :skull:
  * Bokeh is a problem child sometimes, so in case of anything
  * Made a quick script of the steepest descent code. Will make clean and debug.
