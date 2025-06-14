# Compiler and flags
CC = g++
CFLAGS = -Wall -Wextra -std=c++23 -O3

# Executable name
TARGET = main

# Source files and objects
SRCS = $(TARGET).cpp
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Link object files to create the executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Compile source to object file
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(TARGET) $(OBJS)

.PHONY: all clean