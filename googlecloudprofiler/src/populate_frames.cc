#include "populate_frames.h"

#include <Python.h>

#include <errno.h>
#include <fcntl.h>
#include <setjmp.h>
#include <signal.h>
#include <unistd.h>

#include "stacktraces.h"

// Python version definitions
#define PY_311 0x030B0000  // 3.11
#define PY_312 0x030C0000  // 3.12
#define PY_313 0x030D0000  // 3.13

namespace {
// Pipe used by IsReadable() to probe whether memory is mapped without faulting.
// This MUST be a pipe (or socket), NOT /dev/null: the kernel's /dev/null write
// path returns the count without ever doing copy_from_user, so it never detects
// a bad pointer and the probe is useless. A pipe forces copy_from_user.
// Initialized once via InitFramePointerProbe() before profiling starts.
int g_probe_pipe[2] = {-1, -1};
}  // namespace

// Sets up the pipe used for memory-readability probing. Must be called once
// (with the GIL held, before the SIGPROF handler can run) -- see
// Profiler::Reset().
void InitFramePointerProbe() {
  if (g_probe_pipe[1] >= 0) {
    return;
  }
  if (pipe(g_probe_pipe) != 0) {
    g_probe_pipe[0] = g_probe_pipe[1] = -1;
    return;
  }
  // Non-blocking so a transiently full pipe can never block the SIGPROF
  // handler; close-on-exec so the fds don't leak into child processes.
  for (int i = 0; i < 2; i++) {
    int fl = fcntl(g_probe_pipe[i], F_GETFL);
    if (fl != -1) {
      fcntl(g_probe_pipe[i], F_SETFL, fl | O_NONBLOCK);
    }
    fcntl(g_probe_pipe[i], F_SETFD, FD_CLOEXEC);
  }
}

// Returns true iff every byte of [addr, addr+len) is currently readable,
// WITHOUT faulting if it is not.
//
// The frame walk in this file runs inside the SIGPROF handler, which can
// interrupt the interpreter while it is pushing or popping frames (e.g. while a
// data-stack chunk is being munmap'd). In that window the interpreter-internal
// pointers we follow (tstate->cframe->current_frame, frame->previous,
// frame->f_code, ...) can hold non-NULL but invalid values -- empirically a
// small integer such as 0x10, or a stale pointer into freed/unmapped memory. A
// plain `== NULL` check does not catch these, so dereferencing them takes down
// the whole process with SIGSEGV.
//
// write(2) to a pipe forces the kernel to copy_from_user, which returns
// -1/EFAULT (rather than raising SIGSEGV) if any byte is unmapped; write()/read()
// are on the POSIX async-signal-safe list. We immediately drain whatever we
// wrote so the pipe never fills up.
static inline bool IsReadable(const void *addr, size_t len) {
  if (addr == NULL || g_probe_pipe[1] < 0) {
    return false;
  }
  ssize_t w;
  do {
    w = write(g_probe_pipe[1], addr, len);
  } while (w < 0 && errno == EINTR);
  if (w > 0) {
    char buf[256];
    ssize_t drained = 0;
    while (drained < w) {
      ssize_t r;
      do {
        r = read(g_probe_pipe[0], buf, sizeof(buf));
      } while (r < 0 && errno == EINTR);
      if (r <= 0) {
        break;
      }
      drained += r;
    }
  }
  return w == static_cast<ssize_t>(len);
}

// ---------------------------------------------------------------------------
// Fault-protected frame walk.
//
// The readability probe above guards the frame *struct*, but the walk also
// dereferences pointers nested inside CPython inlines that we call as black
// boxes -- e.g. _PyFrame_IsIncomplete() follows frame->f_code to the code
// object, and PyCode_Addr2Line() follows code->co_linetable. During the
// SIGPROF-vs-frame-teardown race any of those can be garbage at arbitrary
// depth, so per-pointer probing cannot cover them all.
//
// We therefore run the whole walk inside a sigsetjmp region with a temporary
// SIGSEGV/SIGBUS handler: any faulting dereference, at any depth, siglongjmps
// back out and the walk is abandoned (0 frames) instead of crashing the
// process. A fault that is NOT inside our walk restores the previous handler
// and returns, so genuine crashes still terminate (and core-dump) normally.
// ---------------------------------------------------------------------------
namespace {
thread_local sigjmp_buf g_walk_jmp;
thread_local volatile sig_atomic_t g_walk_active = 0;
struct sigaction g_old_segv;
struct sigaction g_old_bus;
bool g_guard_installed = false;

void FrameWalkFaultHandler(int sig) {
  if (g_walk_active) {
    g_walk_active = 0;
    siglongjmp(g_walk_jmp, 1);
  }
  // Not a fault inside our frame walk: restore the previous disposition so the
  // re-executed instruction crashes (and core-dumps) as it normally would.
  sigaction(sig, (sig == SIGBUS) ? &g_old_bus : &g_old_segv, NULL);
}
}  // namespace

// Installs the SIGSEGV/SIGBUS guard used by PopulateFramesGuarded(). Call once,
// with the GIL held, before the SIGPROF handler can run (see Profiler::Reset()).
void InstallFrameWalkFaultGuard() {
  if (g_guard_installed) {
    return;
  }
  struct sigaction sa = {};
  sa.sa_handler = FrameWalkFaultHandler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_NODEFER;
  if (sigaction(SIGSEGV, &sa, &g_old_segv) == 0 &&
      sigaction(SIGBUS, &sa, &g_old_bus) == 0) {
    g_guard_installed = true;
  }
}

// Runs PopulateFrames() under the fault-protected region. If any dereference in
// the walk faults, the walk is abandoned and 0 frames are returned instead of
// taking down the process.
int PopulateFramesGuarded(CallFrame *frames, PyThreadState *ts) {
  if (!g_guard_installed) {
    return PopulateFrames(frames, ts);
  }
  int num_frames;
  g_walk_active = 1;
  if (sigsetjmp(g_walk_jmp, 1) == 0) {
    num_frames = PopulateFrames(frames, ts);
  } else {
    // A dereference inside the walk faulted; recover with no frames.
    num_frames = 0;
  }
  g_walk_active = 0;
  return num_frames;
}

#if PY_VERSION_HEX >= PY_313

/**
 * Python 3.13 introduced significant changes to the frame structure:
 * - f_code renamed to f_executable (now PyObject* instead of PyCodeObject*)
 * - prev_instr renamed to instr_ptr
 * - Must use _PyFrame_GetCode() helper to access code object
 *
 * The PyFrameObject structure members have been removed from the public C API
 * in 3.11:
 * https://docs.python.org/3/whatsnew/3.11.html#pyframeobject-3-11-hiding.
 *
 * Since this code runs as part of the SIGPROF handler, it cannot modify Python
 * objects (including their refcounts) and standard getters can't be used.
 * We expose the internal _PyInterpreterFrame and use that directly.
 */

#define Py_BUILD_CORE
#include "internal/pycore_frame.h"
#undef Py_BUILD_CORE

// Modified from CPython 3.13 source for async-signal-safe access
// Python 3.13 flattened cframe->current_frame to just current_frame
//
// IMPORTANT: This can be called from a signal handler (SIGPROF), so we must be
// defensive about race conditions where the interpreter is in the middle of
// setting up or tearing down frames. current_frame / previous can be NULL, or
// non-NULL but pointing at invalid/unmapped memory; every dereference is guarded
// with IsReadable() to avoid segfaulting the process.
static inline _PyInterpreterFrame *unsafe_PyThreadState_GetInterpreterFrame(
    PyThreadState *tstate) {
  if (tstate == NULL) {
    return NULL;
  }

  _PyInterpreterFrame *f = tstate->current_frame;

  // Walk past incomplete frames, validating each pointer is readable before
  // dereferencing it (current_frame / previous may be non-NULL garbage).
  while (IsReadable(f, sizeof(_PyInterpreterFrame)) && _PyFrame_IsIncomplete(f)) {
    f = f->previous;
  }
  if (!IsReadable(f, sizeof(_PyInterpreterFrame))) {
    return NULL;
  }
  return f;
}

// In Python 3.13, f_code became f_executable and is now a PyObject*
// This helper safely extracts the code object
static inline PyCodeObject *unsafe_PyInterpreterFrame_GetCode(
    _PyInterpreterFrame *frame) {
  if (!IsReadable(frame, sizeof(_PyInterpreterFrame)) ||
      _PyFrame_IsIncomplete(frame)) {
    return NULL;
  }

  PyObject *executable = frame->f_executable;
  if (!IsReadable(executable, sizeof(PyObject))) {
    return NULL;
  }

  // f_executable can be a code object or other types; verify it's a code object
  // PyCode_Check uses type pointer, which should be safe to check in signal handler
  if (!PyCode_Check(executable)) {
    return NULL;
  }

  if (!IsReadable(executable, sizeof(PyCodeObject))) {
    return NULL;
  }
  return (PyCodeObject *)executable;
}

static inline _PyInterpreterFrame *unsafe_PyInterpreterFrame_GetBack(
    _PyInterpreterFrame *frame) {
  if (!IsReadable(frame, sizeof(_PyInterpreterFrame)) ||
      _PyFrame_IsIncomplete(frame)) {
    return NULL;
  }

  _PyInterpreterFrame *prev = frame->previous;
  while (IsReadable(prev, sizeof(_PyInterpreterFrame)) &&
         _PyFrame_IsIncomplete(prev)) {
    prev = prev->previous;
  }
  if (!IsReadable(prev, sizeof(_PyInterpreterFrame))) {
    return NULL;
  }
  return prev;
}

// Python 3.13 uses instr_ptr instead of prev_instr
int _PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame) {
  PyCodeObject *code = unsafe_PyInterpreterFrame_GetCode(frame);
  if (code == NULL) {
    return -1;
  }

  int addr = (int)(frame->instr_ptr - _PyCode_CODE(code)) * sizeof(_Py_CODEUNIT);
  return PyCode_Addr2Line(code, addr);
}

int PopulateFrames(CallFrame *frames, PyThreadState *ts) {
  if (ts == nullptr) {
    frames[0].lineno = kNoPyState;
    frames[0].py_code = nullptr;
    return 1;
  }

  _PyInterpreterFrame *frame = unsafe_PyThreadState_GetInterpreterFrame(ts);
  int num_frames = 0;
  while (frame != nullptr && num_frames < kMaxFramesToCapture) {
    // Get code object and line number - might be NULL/-1 if we hit a race condition
    PyCodeObject *code = unsafe_PyInterpreterFrame_GetCode(frame);
    int lineno = _PyInterpreterFrame_GetLine(frame);

    // Only record frames where we successfully got valid data
    // This handles race conditions where frame is partially initialized
    if (code != NULL && lineno >= 0) {
      frames[num_frames].lineno = lineno;
      frames[num_frames].py_code = code;
      num_frames++;
    }

    frame = unsafe_PyInterpreterFrame_GetBack(frame);
  }
  return num_frames;
}

#elif PY_VERSION_HEX >= PY_312

/**
 * Python 3.12 changes to the frame structure:
 * - f_code moved to first position in the struct
 * - f_func renamed to f_funcobj
 * - is_entry field removed
 * - return_offset field added
 *
 * The PyFrameObject structure members have been removed from the public C API
 * in 3.11:
 * https://docs.python.org/3/whatsnew/3.11.html#pyframeobject-3-11-hiding.
 *
 * Since this code runs as part of the SIGPROF handler, it cannot modify Python
 * objects (including their refcounts) and standard getters can't be used.
 * We expose the internal _PyInterpreterFrame and use that directly.
 */

#define Py_BUILD_CORE
#include "internal/pycore_frame.h"
#undef Py_BUILD_CORE

// Modified from CPython 3.12 source for async-signal-safe access
//
// IMPORTANT: This can be called from a signal handler (SIGPROF), which can
// interrupt the Python interpreter at ANY point, including during frame setup
// in _PyEval_EvalFrameDefault. The signal can fire after `tstate->cframe =
// &cframe;` but before `cframe.current_frame = frame;`, so current_frame may be
// NULL or, worse, hold a stale/garbage value. NULL checks alone are not enough
// (we have observed values such as 0x10 and stale pointers into freed memory),
// so every dereference is guarded with IsReadable().
static inline _PyInterpreterFrame *unsafe_PyThreadState_GetInterpreterFrame(
    PyThreadState *tstate) {
  if (tstate == NULL) {
    return NULL;
  }

  // cframe may be NULL during initialization, or non-NULL but invalid.
  _PyCFrame *cframe = tstate->cframe;
  if (!IsReadable(cframe, sizeof(_PyCFrame))) {
    return NULL;
  }

  // current_frame may be NULL or a non-NULL invalid pointer if we interrupted
  // _PyEval_EvalFrameDefault setup. Validate before every dereference.
  _PyInterpreterFrame *f = cframe->current_frame;
  while (IsReadable(f, sizeof(_PyInterpreterFrame)) && _PyFrame_IsIncomplete(f)) {
    f = f->previous;
  }
  if (!IsReadable(f, sizeof(_PyInterpreterFrame))) {
    return NULL;
  }
  return f;
}

// In Python 3.12, f_code is still PyCodeObject* but moved to first position
static inline PyCodeObject *unsafe_PyInterpreterFrame_GetCode(
    _PyInterpreterFrame *frame) {
  if (!IsReadable(frame, sizeof(_PyInterpreterFrame)) ||
      _PyFrame_IsIncomplete(frame)) {
    return NULL;
  }

  PyCodeObject *code = frame->f_code;
  if (!IsReadable(code, sizeof(PyCodeObject))) {
    return NULL;
  }

  return code;
}

static inline _PyInterpreterFrame *unsafe_PyInterpreterFrame_GetBack(
    _PyInterpreterFrame *frame) {
  if (!IsReadable(frame, sizeof(_PyInterpreterFrame)) ||
      _PyFrame_IsIncomplete(frame)) {
    return NULL;
  }

  _PyInterpreterFrame *prev = frame->previous;
  while (IsReadable(prev, sizeof(_PyInterpreterFrame)) &&
         _PyFrame_IsIncomplete(prev)) {
    prev = prev->previous;
  }
  if (!IsReadable(prev, sizeof(_PyInterpreterFrame))) {
    return NULL;
  }
  return prev;
}

// Python 3.12 still uses prev_instr (not renamed yet)
int _PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame) {
  if (!IsReadable(frame, sizeof(_PyInterpreterFrame))) {
    return -1;
  }

  PyCodeObject *code = frame->f_code;
  if (!IsReadable(code, sizeof(PyCodeObject))) {
    return -1;
  }

  int addr = _PyInterpreterFrame_LASTI(frame) * sizeof(_Py_CODEUNIT);
  return PyCode_Addr2Line(code, addr);
}

int PopulateFrames(CallFrame *frames, PyThreadState *ts) {
  if (ts == nullptr) {
    frames[0].lineno = kNoPyState;
    frames[0].py_code = nullptr;
    return 1;
  }

  _PyInterpreterFrame *frame = unsafe_PyThreadState_GetInterpreterFrame(ts);
  int num_frames = 0;
  while (frame != nullptr && num_frames < kMaxFramesToCapture) {
    // Get code object and line number - might be NULL/-1 if we hit a race condition
    PyCodeObject *code = unsafe_PyInterpreterFrame_GetCode(frame);
    int lineno = _PyInterpreterFrame_GetLine(frame);

    // Only record frames where we successfully got valid data
    // This handles race conditions where frame is partially initialized
    if (code != NULL && lineno >= 0) {
      frames[num_frames].lineno = lineno;
      frames[num_frames].py_code = code;
      num_frames++;
    }

    frame = unsafe_PyInterpreterFrame_GetBack(frame);
  }
  return num_frames;
}

#elif PY_VERSION_HEX >= PY_311

/**
 * Python 3.11 frame structure baseline.
 *
 * The PyFrameObject structure members have been removed from the public C API
 * in 3.11:
 * https://docs.python.org/3/whatsnew/3.11.html#pyframeobject-3-11-hiding.
 *
 * Since this code runs as part of the SIGPROF handler, it cannot modify Python
 * objects (including their refcounts) and standard getters can't be used.
 * We expose the internal _PyInterpreterFrame and use that directly.
 */

#define Py_BUILD_CORE
#include "internal/pycore_frame.h"
#undef Py_BUILD_CORE

// Modified from
// https://github.com/python/cpython/blob/v3.11.4/Python/pystate.c#L1278-L1285
//
// IMPORTANT: This can be called from a signal handler (SIGPROF), which can
// interrupt the Python interpreter during frame setup, creating race conditions.
// See Python 3.12 comments above for details on the race condition in
// _PyEval_EvalFrameDefault where cframe is set before current_frame. Every
// dereference is guarded with IsReadable() because the pointers can be non-NULL
// but invalid in that window.
static inline _PyInterpreterFrame *unsafe_PyThreadState_GetInterpreterFrame(
    PyThreadState *tstate) {
  if (tstate == NULL) {
    return NULL;
  }

  _PyCFrame *cframe = tstate->cframe;
  if (!IsReadable(cframe, sizeof(_PyCFrame))) {
    return NULL;
  }

  _PyInterpreterFrame *f = cframe->current_frame;
  while (IsReadable(f, sizeof(_PyInterpreterFrame)) && _PyFrame_IsIncomplete(f)) {
    f = f->previous;
  }
  if (!IsReadable(f, sizeof(_PyInterpreterFrame))) {
    return NULL;
  }
  return f;
}

// Modified from
// https://github.com/python/cpython/blob/v3.11.4/Objects/frameobject.c#L1310-L1315
// with refcounting removed and additional readability checks for signal safety
static inline PyCodeObject *unsafe_PyInterpreterFrame_GetCode(
    _PyInterpreterFrame *frame) {
  if (!IsReadable(frame, sizeof(_PyInterpreterFrame)) ||
      _PyFrame_IsIncomplete(frame)) {
    return NULL;
  }

  PyCodeObject *code = frame->f_code;
  if (!IsReadable(code, sizeof(PyCodeObject))) {
    return NULL;
  }

  return code;
}

// Modified from
// https://github.com/python/cpython/blob/v3.11.4/Objects/frameobject.c#L1326-L1329
// with refcounting removed and additional readability checks for signal safety
static inline _PyInterpreterFrame *unsafe_PyInterpreterFrame_GetBack(
    _PyInterpreterFrame *frame) {
  if (!IsReadable(frame, sizeof(_PyInterpreterFrame)) ||
      _PyFrame_IsIncomplete(frame)) {
    return NULL;
  }

  _PyInterpreterFrame *prev = frame->previous;
  while (IsReadable(prev, sizeof(_PyInterpreterFrame)) &&
         _PyFrame_IsIncomplete(prev)) {
    prev = prev->previous;
  }
  if (!IsReadable(prev, sizeof(_PyInterpreterFrame))) {
    return NULL;
  }
  return prev;
}

// Copied from
// https://github.com/python/cpython/blob/v3.11.4/Python/frame.c#L165-L170 as
// this function is not available in libpython
// Added readability checks for signal safety
int _PyInterpreterFrame_GetLine(_PyInterpreterFrame *frame) {
  if (!IsReadable(frame, sizeof(_PyInterpreterFrame))) {
    return -1;
  }

  PyCodeObject *code = frame->f_code;
  if (!IsReadable(code, sizeof(PyCodeObject))) {
    return -1;
  }

  int addr = _PyInterpreterFrame_LASTI(frame) * sizeof(_Py_CODEUNIT);
  return PyCode_Addr2Line(code, addr);
}

int PopulateFrames(CallFrame *frames, PyThreadState *ts) {
  if (ts == nullptr) {
    frames[0].lineno = kNoPyState;
    frames[0].py_code = nullptr;
    return 1;
  }

  // We are running in the context of the thread interrupted by the signal
  // so the frame object for the current thread is stable.
  // Unfortunately, we can't use PyFrameObjects because they are initialized
  // lazily and will not have the info we need directly.
  _PyInterpreterFrame *frame = unsafe_PyThreadState_GetInterpreterFrame(ts);
  int num_frames = 0;
  while (frame != nullptr && num_frames < kMaxFramesToCapture) {
    // Get code object and line number - might be NULL/-1 if we hit a race condition
    PyCodeObject *code = unsafe_PyInterpreterFrame_GetCode(frame);
    int lineno = _PyInterpreterFrame_GetLine(frame);

    // Only record frames where we successfully got valid data
    // This handles race conditions where frame is partially initialized
    if (code != NULL && lineno >= 0) {
      frames[num_frames].lineno = lineno;
      frames[num_frames].py_code = code;
      num_frames++;
    }

    frame = unsafe_PyInterpreterFrame_GetBack(frame);
  }
  return num_frames;
}

#else
// python versions before 3.11

int PopulateFrames(CallFrame *frames, PyThreadState *ts) {
  if (ts == nullptr) {
    frames[0].lineno = kNoPyState;
    frames[0].py_code = nullptr;
    return 1;
  }
  // We are running in the context of the thread interrupted by the signal
  // so the frame object for the current thread is stable.
  PyFrameObject *frame = ts->frame;
  int num_frames = 0;
  while (frame != nullptr && num_frames < kMaxFramesToCapture) {
    frames[num_frames].lineno = frame->f_lineno;
    frames[num_frames].py_code = frame->f_code;
    num_frames++;
    frame = frame->f_back;
  }
  return num_frames;
}

#endif  // PY_VERSION_HEX >= PY_311
