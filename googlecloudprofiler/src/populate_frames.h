#ifndef THIRD_PARTY_PY_GOOGLECLOUDPROFILER_SRC_POPULATE_FRAMES_H_
#define THIRD_PARTY_PY_GOOGLECLOUDPROFILER_SRC_POPULATE_FRAMES_H_

#include <Python.h>

#include "stacktraces.h"

/**
 * Initializes the resources used to safely walk frames from the SIGPROF
 * handler (currently an fd used for non-faulting memory-readability probes).
 * Must be called once, with the GIL held, before profiling starts.
 */
void InitFramePointerProbe();

/**
 * Installs the SIGSEGV/SIGBUS guard used by PopulateFramesGuarded(). Must be
 * called once, with the GIL held, before the SIGPROF handler can run.
 */
void InstallFrameWalkFaultGuard();

/**
 * Populates the CallFrame array with at-most kMaxFramesToCapture python frames
 * from the provided PyThreadState. Returns the number of frames populated.
 */
int PopulateFrames(CallFrame* frames, PyThreadState* ts);

/**
 * Same as PopulateFrames(), but runs the walk inside a fault-protected region:
 * if any dereference faults (SIGSEGV/SIGBUS) -- e.g. a torn-down frame or code
 * object during the SIGPROF race -- the walk is abandoned and 0 is returned
 * instead of crashing the process. Requires InstallFrameWalkFaultGuard().
 */
int PopulateFramesGuarded(CallFrame* frames, PyThreadState* ts);

#endif  // THIRD_PARTY_PY_GOOGLECLOUDPROFILER_SRC_POPULATE_FRAMES_H_
