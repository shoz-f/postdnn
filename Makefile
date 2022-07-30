# Makefile for building the NIF
#
# MIX_APP_PATH          path to the build directory
# ERL_EI_INCLUDE_DIR    include path to erlang header
# ERL_EI_LIBDIR         path to erlang/c libraries (Not necessaly for NIFs)

ifeq ($(MIX_APP_PATH),)
calling_from_make:
	mix compile
endif

HOSTOS		= $(shell uname -s)
NIF_NAME	= postdnn_nif

# Directories
PRIV		= $(MIX_APP_PATH)/priv
BUILD		= $(MIX_APP_PATH)/obj

# Build options
CFLAGS		+= -O2 -Isrc $(addprefix -I, $(EXTRA_LIB)) -pedantic -fPIC
LDFLAGS		+= -shared
ERL_CFLAGS	?= -I"$(ERL_EI_INCLUDE_DIR)"
ERL_LDFLAGS	?= -L"$(ERL_EI_LIBDIR)"
ifneq (,$(CROSSCOMPILE))
    NIFS = $(PRIV)/$(NIF_NAME).so
    LDFLAGS += -lm -lpthread
    #CFLAGS  += -D

else ifneq (,$(findstring MSYS_NT,$(HOSTOS)))
    NIFS = $(PRIV)/$(NIF_NAME).dll
    ifneq (,$(findstring $(MIX_ENV), dev test))
        #LDFLAGS += -l
    else
        #CFLAGS  += -D
    endif

else ifeq (Linux, $(HOSTOS))
    NIFS = $(PRIV)/$(NIF_NAME).so
    LDFLAGS += -lm -lpthread
    ifneq (,$(findstring $(MIX_ENV), dev test))
        #LDFLAGS += -L/usr/X11R6/lib -lX11
    else
        #CFLAGS  += -Dcimg_display=0
    endif

else
    $(error Not available system "$(HOSTOS)")
endif

# Target list
HDRS = $(wildcard src/*.h)
SRCS = $(wildcard src/*.cc)
OBJS = $(SRCS:src/%.cc=$(BUILD)/%.o)

# Build rules
all: setup build

setup: $(PRIV) $(BUILD)

build: $(NIFS)

$(BUILD)/%.o: src/%.cc $(HDRS) Makefile
	@echo "-CXX $(notdir $@)"
	$(CXX) -c $(ERL_CFLAGS) $(CFLAGS) -o $@ $<

$(NIFS): $(OBJS)
	@echo "-LD $(notdir $@)"
	$(CXX) $^ $(ERL_LDFLAGS) $(LDFLAGS) -o $@

$(PRIV) $(BUILD):
	mkdir -p $@

clean:
	$(RM) $(NIFS) $(BUILD)/*.o src/*.inc lib/postdnn/$(NIF_NAME).ex

.PHONY: all clean setup build

################################################################################
# Download 3rd-party libraries
EXTRA_LIB   += ./3rd_party/nlohmann_json/single_include
./3rd_party/nlohmann_json:
	@echo "-DOWNLOAD $(notdir $@)"
	mkdir -p $(dir $@)
	git clone https://github.com/nlohmann/json.git $@

setup: $(EXTRA_LIB)

################################################################################
# NIF name
NIF_TABLE	= src/postdnn_nif.inc
src/postdnn_nif.inc: src/postdnn_nif.cc
	@echo "-GENERATE $(notdir $@)"
	python3 nif_tbl.py -o $@ --prefix dnn_ $<

NIF_STUB	= lib/postdnn/$(NIF_NAME).ex
$(NIF_STUB): $(NIF_TABLE)
	@echo "-GENERATE $(notdir $@)"
	python3 nif_stub.py -o $@ PostDNN.NIF $^

$(BUILD)/$(NIF_NAME).o: $(NIF_STUB)

# Don't echo commands unless the caller exports "V=1"
${V}.SILENT:
