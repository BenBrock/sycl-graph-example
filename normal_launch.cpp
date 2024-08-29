#include <sycl/sycl.hpp>
#include <vector>
#include <cassert>

namespace sycl_ext = sycl::ext::oneapi::experimental;

void test_data_dependency(sycl::queue& Queue, float* PtrA, float* PtrB, float* PtrC, bool state_dependencies = true, std::size_t N = 128) {
  // A = 1, B = 2, C = 3
  auto InitEvent = Queue.submit([&](auto&& CGH) {
    CGH.parallel_for(N, [=](auto id) {
      PtrA[id] = 1.0f;
      PtrB[id] = 2.0f;
      PtrC[id] = 3.0f;
    });
  });

  const std::size_t n_launches = 10;

  std::vector<sycl::event> a_events;
  a_events.reserve(n_launches);

  // A += 1, n_launches times.

  for (std::size_t i = 0; i < n_launches; i++) {
    auto EventA = Queue.submit([&](auto&& CGH) {
      if (state_dependencies) {
        if (a_events.empty()) {
          CGH.depends_on(InitEvent);
        } else {
          CGH.depends_on(a_events.back());
        }
      }
      CGH.parallel_for(N, [=](auto id) {
        PtrA[id] += 1.0f;
      });
    });
    a_events.push_back(EventA);
  }



  std::vector<sycl::event> b_events;
  b_events.reserve(n_launches);

  // B += 1.5, n_launches times.

  for (std::size_t i = 0; i < n_launches; i++) {
    auto EventB = Queue.submit([&](auto&& CGH) {
      if (state_dependencies) {
        if (b_events.empty()) {
          CGH.depends_on(InitEvent);
        } else {
          CGH.depends_on(b_events.back());
        }
      }
      CGH.parallel_for(N, [=](auto id) {
        PtrB[id] += 1.5f;
      });
    });
    b_events.push_back(EventB);
  }

  // C += (A + B)
  auto EventC = Queue.submit([&](auto&& CGH) {
    if (state_dependencies) {
      CGH.depends_on(a_events);
      CGH.depends_on(b_events);
    }
    CGH.parallel_for(N, [=](auto id) {
      PtrC[id] += (PtrA[id] + PtrB[id]);
    });
  });
   
  Queue.wait_and_throw();

  std::vector<float> c_local(N);

  float value = 1.0 + 1.0*n_launches + 2.0 + 1.5*n_launches + 3.0;

  Queue.memcpy(c_local.data(), PtrC, sizeof(float)*N).wait_and_throw();

  constexpr bool print_value = false;

  for (std::size_t i = 0; i < c_local.size(); i++) {
    if (print_value && c_local[i] != value) {
      printf("%f (test) != %f (truth)\n", c_local[i], value);
    }
    assert(c_local[i] == value);
  }
}

int main(int argc, char** argv) {
  std::size_t N = 128*1024;
  std::size_t num_trials = 1000;
   
  sycl::queue Queue(sycl::gpu_selector_v);

  float* PtrA = sycl::malloc_device<float>(N, Queue);
  float* PtrB = sycl::malloc_device<float>(N, Queue);
  float* PtrC = sycl::malloc_device<float>(N, Queue);

  assert(PtrA != nullptr);
  assert(PtrB != nullptr);
  assert(PtrC != nullptr);

  printf("Normal kernel launches.\n");
  printf("Testing *with* data dependencies stated. This should not fail.\n");
  fflush(stdout);

  for (std::size_t i = 0; i < num_trials; i++) {
    test_data_dependency(Queue, PtrA, PtrB, PtrC, true, N);
  }

  printf("Testing *without* data dependencies. This should throw an assertion error...\n");
  fflush(stdout);

  for (std::size_t i = 0; i < num_trials; i++) {
    test_data_dependency(Queue, PtrA, PtrB, PtrC, false, N);
  }


  return 0;
}
