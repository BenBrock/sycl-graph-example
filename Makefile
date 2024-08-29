DPCXX = icpx

SOURCES += $(wildcard *.cpp)
DPCXX_TARGETS := $(patsubst %.cpp, %, $(SOURCES))
TARGETS := $(DPCXX_TARGETS)

CXXFLAGS = -std=c++23 -O3

DPCPP_FLAGS = -fsycl

all: $(TARGETS)

run: all
	@for target in $(foreach target,$(TARGETS),./$(target)) ; do echo "Running \"$$target\"" ; $$target ; done


dpcpp: $(DPCXX_TARGETS)

%: %.cpp
	$(DPCXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS) $(DPCPP_FLAGS) $(LDLIBS)

clean:
	rm -fv $(TARGETS)
