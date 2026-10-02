ACLOCAL_AMFLAGS = -I m4

# 这里就算标记了 -O3， 会被后面 configure 中 gcc 编译FLAG 覆盖 -O2
AM_CFLAGS = -march=native -I$(top_srcdir)/include -I$(top_srcdir)/src $(DEPS_CFLAGS)

dist_doc_DATA = COPYING AUTHORS README

include_HEADERS = include/rnnoise.h

lib_LTLIBRARIES = librnnoise.la
noinst_HEADERS = src/arch.h  \
		 src/celt_lpc.h  \
		 src/cpu_support.h \
		 src/common.h  \
		 src/denoise.h \
		 src/_kiss_fft_guts.h  \
		 src/kiss_fft.h  \
		 src/nnet.h \
		 src/nnet_arch.h \
		 src/opus_types.h  \
		 src/pitch.h  \
		 src/rnn.h  \
		 src/rnnoise_data.h \
		 src/vec_neon.h \
		 src/vec.h \
		 src/vec_avx.h \
		 src/x86/x86_arch_macros.h \
		 src/x86/x86cpu.h \
		 src/x86/dnn_x86.h \
		 examples/rnn_unit.h \
		 examples/rnn_unit_util.h

RNNOISE_SOURCES = \
	src/denoise.c \
	src/rnn.c \
	src/pitch.c \
	src/kiss_fft.c \
	src/celt_lpc.c \
	src/nnet.c \
	src/nnet_default.c \
	src/parse_lpcnet_weights.c \
	src/rnnoise_data.c \
	src/rnnoise_tables.c

RNNOISE_SOURCES_SSE4_1 = src/x86/nnet_sse4_1.c
RNNOISE_SOURCES_AVX2 = src/x86/nnet_avx2.c

X86_RTCD = src/x86/x86_dnn_map.c \
	   src/x86/x86cpu.c

if RNN_ENABLE_X86_RTCD
RNNOISE_SOURCES += $(X86_RTCD) $(RNNOISE_SOURCES_SSE4_1) $(RNNOISE_SOURCES_AVX2)
endif

librnnoise_la_SOURCES = $(RNNOISE_SOURCES)
librnnoise_la_LIBADD = $(DEPS_LIBS) $(lrintf_lib) $(LIBM)
librnnoise_la_LDFLAGS = -no-undefined \
 -version-info @OP_LT_CURRENT@:@OP_LT_REVISION@:@OP_LT_AGE@

noinst_PROGRAMS = dump_features dump_weights_blob dump_rnn_io rnn_unit
if OP_ENABLE_EXAMPLES
noinst_PROGRAMS += examples/rnnoise_demo
endif

examples_rnnoise_demo_SOURCES = examples/rnnoise_demo.c
examples_rnnoise_demo_LDADD = librnnoise.la

dump_features_SOURCES = src/dump_features.c src/denoise.c src/pitch.c src/celt_lpc.c src/kiss_fft.c src/parse_lpcnet_weights.c src/rnnoise_tables.c
dump_features_LDADD = $(LIBM)
dump_features_CFLAGS = $(AM_CFLAGS) -DTRAINING

dump_weights_blob_SOURCES = src/write_weights.c
dump_weights_blob_LDADD = $(LIBM)
dump_weights_blob_CFLAGS = $(AM_CFLAGS) -DDUMP_BINARY_WEIGHTS

dump_rnn_io_SOURCES = src/denoise.c src/kiss_fft.c src/pitch.c src/celt_lpc.c src/rnn.c src/nnet.c src/nnet_default.c \
         src/parse_lpcnet_weights.c src/rnnoise_tables.c src/rnnoise_data.c src/dump_rnn_io.c \
		 src/x86/x86_dnn_map.c src/x86/x86cpu.c
dump_rnn_io_LDADD = $(LIBM)
dump_rnn_io_CFLAGS = $(AM_CFLAGS)

# 精度开关（刻意只加到 rnn_unit_CFLAGS，不动全局 AM_CFLAGS）：
#   空（默认）                              -> 低精度：多项式近似 + _mm256_rcp_ps
#   RNN_UNIT_HIGH_ACCURACY=-DHIGH_ACCURACY  -> 高精度：1.f/(1+exp(-x)) / tanhf
# 为什么够用：automake 会把 src/nnet_default.c 编成 per-target 对象
# src/rnn_unit-nnet_default.o，它的编译行用的就是 $(rnn_unit_CFLAGS)；
# 而 HIGH_ACCURACY 正是在 nnet_default.c 包含 src/nnet_arch.h 时（第 77 行）生效。
RNN_UNIT_HIGH_ACCURACY =
rnn_unit_SOURCES = examples/rnn_unit.c examples/rnn_unit_util.c src/nnet.c src/x86/x86_dnn_map.c src/nnet_default.c
rnn_unit_LDADD = $(LIBM)
rnn_unit_CFLAGS = $(AM_CFLAGS) -g $(RNN_UNIT_HIGH_ACCURACY)


pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = rnnoise.pc

debug:
	$(MAKE) CFLAGS="${CFLAGS} -O0 -ggdb -DOP_ENABLE_ASSERTIONS" all

EXTRA_DIST = \
 rnnoise.pc.in \
 rnnoise-uninstalled.pc.in \
 doc/Doxyfile.in \
 doc/Makefile

# Targets to build and install just the library without the docs
librnnoise install-librnnoise: NO_DOXYGEN = 1

rnnoise: all
install-rnnoise: install

# Or just the docs
docs: doc/doxygen-build.stamp

install-docs:
	@if [ -z "$(NO_DOXYGEN)" ]; then \
	  ( cd doc && \
	    echo "Installing documentation in $(DESTDIR)$(docdir)"; \
	    $(INSTALL) -d $(DESTDIR)$(docdir)/html/search; \
	    for f in `find html -type f \! -name "installdox"` ; do \
	      $(INSTALL_DATA) $$f $(DESTDIR)$(docdir)/$$f;     \
	    done ) \
	fi

doc/doxygen-build.stamp: doc/Doxyfile \
                         $(top_srcdir)/include/*.h
	@[ -n "$(NO_DOXYGEN)" ] || ( cd doc && doxygen && touch $(@F) )


if HAVE_DOXYGEN

# Or everything (by default)
all-local: docs

install-data-local: install-docs

clean-local:
	$(RM) -r doc/html
	$(RM) -r doc/latex
	$(RM) doc/doxygen-build.stamp

uninstall-local:
	$(RM) -r $(DESTDIR)$(docdir)/html

endif

# We check this every time make is run, with configure.ac being touched to
# trigger an update of the build system files if update_version changes the
# current PACKAGE_VERSION (or if package_version was modified manually by a
# user with either AUTO_UPDATE=no or no update_version script present - the
# latter being the normal case for tarball releases).
#
# We can't just add the package_version file to CONFIGURE_DEPENDENCIES since
# simply running autoconf will not actually regenerate configure for us when
# the content of that file changes (due to autoconf dependency checking not
# knowing about that without us creating yet another file for it to include).
#
# The MAKECMDGOALS check is a gnu-make'ism, but will degrade 'gracefully' for
# makes that don't support it.  The only loss of functionality is not forcing
# an update of package_version for `make dist` if AUTO_UPDATE=no, but that is
# unlikely to be a real problem for any real user.
$(top_srcdir)/configure.ac: force
	@case "$(MAKECMDGOALS)" in \
	    dist-hook)                             exit 0       ;; \
	    dist-* | dist | distcheck | distclean) _arg=release ;; \
	esac; \
	if ! $(top_srcdir)/update_version $$_arg 2> /dev/null; then \
	    if [ ! -e $(top_srcdir)/package_version ]; then \
		echo 'PACKAGE_VERSION="unknown"' > $(top_srcdir)/package_version; \
	    fi; \
	    . $(top_srcdir)/package_version || exit 1; \
	    [ "$(PACKAGE_VERSION)" != "$$PACKAGE_VERSION" ] || exit 0; \
	fi; \
	touch $@

force:

# Create a minimal package_version file when make dist is run.
dist-hook:
	echo 'PACKAGE_VERSION="$(PACKAGE_VERSION)"' > $(top_distdir)/package_version


.PHONY: rnnoise install-rnnoise docs install-docs

if RNN_ENABLE_X86_RTCD
SSE4_1_OBJ = $(RNNOISE_SOURCES_SSE4_1:.c=.lo)
$(SSE4_1_OBJ): CFLAGS += $(OPUS_X86_SSE4_1_CFLAGS)

AVX2_OBJ = $(RNNOISE_SOURCES_AVX2:.c=.lo)
$(AVX2_OBJ): CFLAGS += $(OPUS_X86_AVX2_CFLAGS)
endif
