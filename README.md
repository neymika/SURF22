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
  ## :sweat: :broken_heart: Week 5...
    * 7/11: Kept trying to debug code using toysvrg.py from Week 4. Determined that some of the convergence rate comparisons made with GD were different when GD used the backtracking line search method to determine the step size vs. the SGD step size schedule. Read a 42 (more like 33) page survey paper of multi-fidelity methods. Very interesting! Learned a lot about the different types of low-fidelity models (simplified, projection-based, and data-fit models), model management strategies (adaptation, fusion, and filtering), and applications (uncertainty propagation, statistical inference, and optimization).
    * 7/12: Discussed the survey paper. Many questions were answered. Unfortunately, I still have the issue where I fail to phrase things well. Be it in discussion or writing :( Honestly its very disheartening that I cannot seem to improve in this aspect which is vital in industry and grad school... In any case, I am implementing a line search method for the stochastic methods to see if I can make more viable comparisons to GD with backtracking.
      * I did a thing! Managed to implement the line search but the results are very confusing. ~~Now trying to plot~~ Managed to plot L(w) vs. grad psi_i/n (from algorithm and line search) + psi_i/n (from line search). While I recognize that psi_i may not have the same computational cost as grad psi_i, I think this is an interesting comparison to make by empirically approximating the two to be equal. Later I will try to make them equal theoretically as well. ** UPDATE ** Managed to plot, but no real added info :sob:
      * To run sgd + gd with backtracking files:
        * ```python
          cd Week5
          python3 toysvrg.py
          ```
    * 7/13: Just keep swimming, just keep swimming... Since the backtracking results were less than ideal to put it mildly, we are switching back to GD having the same step size schedule as SGD and then comparing. Going to use Pan's optimal values and then compare with SGD and GD to see if performance improves, which would indicate there is some sort of tradeoff. Later, if this works, would finetune by controlling all parameters than varying the SVRG batch size parameters only in btwn the two.
      * ** UPDATE ** Compared with SGD and GD with step size dependent on prior scheduling method. Works EXACTLY as intended woooooot (see toy_idealparam.png). Going to try and optimize the constant step size parameter for SGD and GD next then compare.
      * To run Pan optimized SVRG method:
        * ```python
          cd Week5
          python3 toypan.py
          ```
    * 7/14 Instead of trying to optimize the constant step size parameter alone then number of iterations/epochs the constant step size is used for SGD and GD, I am using a grid based approach albeit with less accuracy to determine better step size schedule parameters. I tried paralleling the process for GD using ray, but perhaps due to memory spilling and/or limited CPU capability, the results seemed bizarre and perhaps inaccurate. As such, I am running the grid search sequentially instead, rerunning the grid search in parallel, and then comparing the results. All in all, the sequential method is veryyyyy slow compared to the parallel version (~~has been running for about 1.5 hours~~ Took 4 hours to run! Parallel took 3... Redid with smaller range of fixed eta epochs (1-20)). I wish Pan would respond back so I can ask for help, but I know he has other priorities now that he is in Beijing :persevere:
      * For proper comparisons of Pan parameters for all methods, please see toy_panparam.png
      * For proper comparisons of SVRG with "ideal" Pan parameters v.s. optimal fixed scheduling methods, please see toy_idealnewparam.png
  ## :pensive: :sweat_drops: Week 6
    * 7/18 Reran the ideal SGD with better optimal parameter checking conditions (avoid overflow issues and not use terminating condition necessarily). Plan for today is to produce plots of L vs t and ||nabla L(w)|| vs t for both SGD and GD when varying both parameters logarithmically and then comparing within methods rather than between methods. ** UPDATE ** Done!
      * To run the code to produce the plots for SGD method:
        * ```python
          cd Week6
          python3 sgdvaried.py
          python3 sgdvariedplots.py
          ```
      * To run the code to produce the plots for SGD method:
        * ```python
        cd Week6
        python3 sgdvaried.py
        python3 sgdvariedplots.py
        ```
    *
