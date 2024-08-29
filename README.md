# README

This repo contains two very basic examples that launch kernels with dependencies.

`normal_launch` uses normal SYCL kernel launches, which are asynchronous, but in general greedily launched by the OpenCL and Level Zero runtimes.

`graph_launch` compiles all these kernels into a single SYCL graph that can be launched more efficiently on the GPU.  It requires the Level Zero runtime, since the OpenCL backend currently does not have SYCL graph support.

These examples launch a bunch of chained kernels which depend on one another.  They perform floating point accumulations that must be ordered in order to avoid race conditions.

Each example first launches a bunch of kernels *with* the dependencies specified.  This should execute correctly and throw no assertions.  Finally, the kernels will be launched *without* any dependencies specified.  This should throw an assertion, assuming your implementation supports asynchrony.

Executing on an Intel Ponte Vecchio GPU with oneAPI 2024.2.1 configured to use the default Level Zero backend, I see the following output:

```bash
(base) bbrock@sdp699058:~/src/sycl-doodles$ make run
icpx -std=c++23 -O3 -o graph_launch graph_launch.cpp  -fsycl
icpx -std=c++23 -O3 -o normal_launch normal_launch.cpp  -fsycl
Running "./graph_launch"
Normal kernel launches.
Testing *with* data dependencies stated. This should not fail.
Testing *without* data dependencies. This should throw an assertion error...
graph_launch: graph_launch.cpp:95: auto execute_graph(sycl::queue &, auto &&, float *, float *, float *, bool, std::size_t) [graph:auto = sycl::ext::oneapi::experimental::command_graph<graph_state::executable> &]: Assertion `c_local[i] == value' failed.
Aborted (core dumped)
Running "./normal_launch"
Normal kernel launches.
Testing *with* data dependencies stated. This should not fail.
Testing *without* data dependencies. This should throw an assertion error...
normal_launch: normal_launch.cpp:88: void test_data_dependency(sycl::queue &, float *, float *, float *, bool, std::size_t): Assertion `c_local[i] == value' failed.
Aborted (core dumped)
make: *** [Makefile:14: run] Error 134
```

This is as expected: the kernels are executed in the correct order when dependencies are specified, both when using the normal eager launch method and with SYCL graphs.  When dependencies are not specified, the test fails.
