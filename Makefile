PROG := helium_hf
SRCS :=	helium_hf.cpp

OBJS = helium_hf.o
DEPS = helium_hf.d

VPATH  = src
CXX = clang++
CXXFLAGS = -Wall -Wextra -O3 -std=c++17 -mtune=native -march=native
LDFLAGS = -lm -ldl

all: $(PROG) ;
#rm -f $(OBJS) $(DEPS)

-include $(DEPS)

$(PROG): $(OBJS)
		$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
		$(CXX) $(CXXFLAGS) -c -MMD -MP $<

clean:
		rm -f $(PROG) $(OBJS) $(DEPS)
