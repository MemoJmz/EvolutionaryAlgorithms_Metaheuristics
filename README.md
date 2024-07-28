# Estimation of Distribution Algorithm

The Estimation of Distribution Algorithm (EDA) is a metaheuristic method, i.e., the EDA is a method to estimate local minima of a objective function. It belongs to a Evolutionary Algorithms, see [Wiki](https://en.wikipedia.org/wiki/Estimation_of_distribution_algorithm) & ![Fig.](https://en.wikipedia.org/wiki/File:Eda_mono-variant_gauss_iterations.svg).

I present an implementation of the method and its application. The implementation is based of the next pseudocode:

	0. t := 0
	1. initialize model M(0) to represent uniform distribution 	over admissible solutions
	2. while (termination criteria not met) do:
		2(a). P := generate N>0 candidate solutions by sampling M(t)
		2(b). F := evaluate all candidate solutions in P
    M(t + 1) := adjust_model(P, F, M(t))
	    2(c). t := t + 1

# Files

- The all code is contained in the file: ***proyecto.cpp***
- A easy integration of parallel computing: ***proyecto_par.cpp***
- A full description of the application is on the file [in Spanish]:  ***DescriptionResults_Project.pdf***
- Code for experiments with benchmark fucntions: ***proyecto_benchmark.cpp***
- To plot the benchmark functions we use python: ***Project_CP_MetaHeu.ipynb***
