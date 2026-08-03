Profiling
=========

To profile the code do the following:

1. Add the `-pg` flag to the compiler
2. Run the executable `my_progam`
3. Run gprof `gprof my_progam gmon.out > analysis.txt`
4. Take a look at `analysis.txt`


How to interpret profiling analysis
-----------------------------------

- Take a look at the `% time` column. It indicates which function the program spends most time in. This is likely the bottleneck of the program.
- Take a look at the `calls` column. If a function is called a lot of times, it might be worth to speed it up



Improvements
------------
- I tried using the numerical approximation of the Voigt function used in the Zaytsev paper, but that is much slower than using the Fadeeva function
