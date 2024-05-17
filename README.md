# CUDA_Matrix_Multiplication
Applied Parallel Programming on GPU
Parallelization via CUDA in matrix multiplication offers a powerful means to expedite 
computation-intensive tasks. In the problem domain of matrix multiplication, represented by two 
matrices 𝐴 and 𝐵 of sizes 𝑀×𝑁 and 𝑁×𝑃 respectively, the conventional sequential approach 
entails computing each element Cij individually. However, leveraging CUDA parallelization, 
each thread computes one element of the output matrix C. By organizing threads into blocks and 
grids, and distributing the workload across multiple threads, CUDA enables concurrent execution 
of independent computations. This parallel execution results in a substantial reduction in the 
overall computation time, making it feasible to handle larger matrices efficiently. The time 
complexity of the parallelized approach scales favorably with the total number of threads, 
offering improved performance for large matrix sizes. Consequently, CUDA parallelization 
transforms matrix multiplication into a highly scalable and efficient operation, essential for 
various scientific and engineering applications where rapid computation of matrix products are
paramount.
