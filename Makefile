CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -D_POSIX_C_SOURCE=199309L
LDFLAGS = -lm

SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c,$(OBJDIR)/%.o,$(SOURCES))
TARGET = $(BINDIR)/matrix_profile

.PHONY: all clean dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(OBJDIR) $(BINDIR)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) -I$(INCDIR) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

# Run the profiler
run: all
	./$(TARGET)

# Run with profiling script
profile: all
	./profile.sh
