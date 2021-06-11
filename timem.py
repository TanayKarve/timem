from functools import wraps, partial
from asyncio import coroutine, iscoroutinefunction
from contextlib import contextmanager
import subprocess
import time
import inspect
import os
import warnings
import sys
import linecache
import psutil
import pdb

try:
    import tracemalloc

    has_tracemalloc = True
except ImportError:
    has_tracemalloc = False

_TWO_20 = float(2 ** 20)


def show_results(prof, stream=None, precision=1):
    if stream is None:
        stream = sys.stdout
    template = "{0:>6} {1:>12} {2:>12}  {3:>10}   {4:<}"

    for (filename, lines) in prof.code_map.items():
        header = template.format(
            "Line #", "Mem usage", "Increment", "Occurences", "Line Contents"
        )

        stream.write(u"Filename: " + filename + "\n\n")
        stream.write(header + u"\n")
        stream.write(u"=" * len(header) + "\n")

        all_lines = linecache.getlines(filename)

        float_format = u"{0}.{1}f".format(precision + 4, precision)
        template_mem = u"{0:" + float_format + "} MiB"
        for (lineno, mem) in lines:
            if mem:
                inc = mem[0]
                total_mem = mem[1]
                total_mem = template_mem.format(total_mem)
                occurences = mem[2]
                inc = template_mem.format(inc)
            else:
                total_mem = u""
                inc = u""
                occurences = u""
            tmp = template.format(
                lineno, total_mem, inc, occurences, all_lines[lineno - 1]
            )
            stream.write(tmp)
        stream.write(u"\n\n")


def _get_child_memory(process, meminfo_attr=None, memory_metric=0):
    """
    Returns a generator that yields memory for all child processes.
    """
    # Convert a pid to a process
    if isinstance(process, int):
        if process == -1:
            process = os.getpid()
        process = psutil.Process(process)

    if not meminfo_attr:
        # Use the psutil 2.0 attr if the older version isn't passed in.
        meminfo_attr = (
            "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
        )

    # Select the psutil function get the children similar to how we selected
    # the memory_info attr (a change from excepting the AttributeError).
    children_attr = "children" if hasattr(process, "children") else "get_children"

    # Loop over the child processes and yield their memory
    try:
        for child in getattr(process, children_attr)(recursive=True):
            if isinstance(memory_metric, str):
                meminfo = getattr(child, meminfo_attr)()
                yield getattr(meminfo, memory_metric) / _TWO_20
            else:
                yield getattr(child, meminfo_attr)()[memory_metric] / _TWO_20
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        # https://github.com/fabianp/memory_profiler/issues/71
        yield 0.0


def _get_memory(pid, backend, timestamps=False, include_children=False, filename=None):
    # .. low function to get memory consumption ..
    if pid == -1:
        pid = os.getpid()

    def tracemalloc_tool():
        # .. cross-platform but but requires Python 3.4 or higher ..
        stat = next(
            filter(
                lambda item: str(item).startswith(filename),
                tracemalloc.take_snapshot().statistics("filename"),
            )
        )
        mem = stat.size / _TWO_20
        if timestamps:
            return mem, time.time()
        else:
            return mem

    def ps_util_tool():
        # .. cross-platform but but requires psutil ..
        process = psutil.Process(pid)
        try:
            # avoid using get_memory_info since it does not exists
            # in psutil > 2.0 and accessing it will cause exception.
            meminfo_attr = (
                "memory_info" if hasattr(process, "memory_info") else "get_memory_info"
            )
            mem = getattr(process, meminfo_attr)()[0] / _TWO_20
            if include_children:
                mem += sum(_get_child_memory(process, meminfo_attr))
            if timestamps:
                return mem, time.time()
            else:
                return mem
        except psutil.AccessDenied:
            pass
            # continue and try to get this from ps

    def _ps_util_full_tool(memory_metric):

        # .. cross-platform but requires psutil > 4.0.0 ..
        process = psutil.Process(pid)
        try:
            if not hasattr(process, "memory_full_info"):
                raise NotImplementedError(
                    "Backend `{}` requires psutil > 4.0.0".format(memory_metric)
                )

            meminfo_attr = "memory_full_info"
            meminfo = getattr(process, meminfo_attr)()

            if not hasattr(meminfo, memory_metric):
                raise NotImplementedError(
                    "Metric `{}` not available. For details, see:".format(memory_metric)
                    + "https://psutil.readthedocs.io/en/latest/index.html?highlight=memory_info#psutil.Process.memory_full_info"
                )
            mem = getattr(meminfo, memory_metric) / _TWO_20

            if include_children:
                mem += sum(_get_child_memory(process, meminfo_attr, memory_metric))

            if timestamps:
                return mem, time.time()
            else:
                return mem

        except psutil.AccessDenied:
            pass
            # continue and try to get this from ps

    def posix_tool():
        # .. scary stuff ..
        if include_children:
            raise NotImplementedError(
                (
                    "The psutil module is required to monitor the "
                    "memory usage of child processes."
                )
            )

        warnings.warn("psutil module not found. memory_profiler will be slow")
        # ..
        # .. memory usage in MiB ..
        # .. this should work on both Mac and Linux ..
        # .. subprocess.check_output appeared in 2.7, using Popen ..
        # .. for backwards compatibility ..
        out = (
            subprocess.Popen(["ps", "v", "-p", str(pid)], stdout=subprocess.PIPE)
            .communicate()[0]
            .split(b"\n")
        )
        try:
            vsz_index = out[0].split().index(b"RSS")
            mem = float(out[1].split()[vsz_index]) / 1024
            if timestamps:
                return mem, time.time()
            else:
                return mem
        except:
            if timestamps:
                return -1, time.time()
            else:
                return -1

    if backend == "tracemalloc" and (filename is None or filename == "<unknown>"):
        raise RuntimeError("There is no access to source file of the profiled function")

    tools = {
        "tracemalloc": tracemalloc_tool,
        "psutil": ps_util_tool,
        "psutil_pss": lambda: _ps_util_full_tool(memory_metric="pss"),
        "psutil_uss": lambda: _ps_util_full_tool(memory_metric="uss"),
        "posix": posix_tool,
    }
    return tools[backend]()


class CodeMap(dict):
    def __init__(self, include_children, backend):
        self.include_children = include_children
        self._toplevel = []
        self.backend = backend

    def add(self, code, toplevel_code=None):
        if code in self:
            return

        if toplevel_code is None:
            filename = code.co_filename
            if filename.endswith((".pyc", ".pyo")):
                filename = filename[:-1]
            if not os.path.exists(filename):
                print("ERROR: Could not find file " + filename)
                if filename.startswith(("ipython-input", "<ipython-input")):
                    print(
                        "NOTE: %mprun can only be used on functions defined in"
                        " physical files, and not in the IPython environment."
                    )
                return

            toplevel_code = code
            (sub_lines, start_line) = inspect.getsourcelines(code)
            linenos = range(start_line, start_line + len(sub_lines))
            self._toplevel.append((filename, code, linenos))
            self[code] = {}
        else:
            self[code] = self[toplevel_code]

        for subcode in filter(inspect.iscode, code.co_consts):
            self.add(subcode, toplevel_code=toplevel_code)

    def trace(self, code, lineno, prev_lineno):
        memory = _get_memory(
            -1,
            self.backend,
            include_children=self.include_children,
            filename=code.co_filename,
        )
        prev_value = self[code].get(lineno, None)
        previous_memory = prev_value[1] if prev_value else 0
        previous_inc = prev_value[0] if prev_value else 0

        prev_line_value = self[code].get(prev_lineno, None) if prev_lineno else None
        prev_line_memory = prev_line_value[1] if prev_line_value else 0
        occ_count = self[code][lineno][2] + 1 if lineno in self[code] else 1
        self[code][lineno] = (
            previous_inc + (memory - prev_line_memory),
            max(memory, previous_memory),
            occ_count,
        )

    def items(self):
        """Iterate on the toplevel code blocks."""
        for (filename, code, linenos) in self._toplevel:
            measures = self[code]
            if not measures:
                continue  # skip if no measurement
            line_iterator = ((line, measures.get(line)) for line in linenos)
            yield (filename, line_iterator)


class LineProfiler(object):
    """ A profiler that records the amount of memory for each line """

    def __init__(self, **kw):
        include_children = kw.get("include_children", False)
        backend = kw.get("backend", "psutil")
        self.code_map = CodeMap(include_children=include_children, backend=backend)
        self.enable_count = 0
        self.max_mem = kw.get("max_mem", None)
        self.prevlines = []
        self.backend = choose_backend(kw.get("backend", None))
        self.prev_lineno = None

    def __call__(self, func=None, precision=1):
        if func is not None:
            self.add_function(func)
            f = self.wrap_function(func)
            f.__module__ = func.__module__
            f.__name__ = func.__name__
            f.__doc__ = func.__doc__
            f.__dict__.update(getattr(func, "__dict__", {}))
            return f
        else:

            def inner_partial(f):
                return self.__call__(f, precision=precision)

            return inner_partial

    def add_function(self, func):
        """Record line profiling information for the given Python function."""
        try:
            # func_code does not exist in Python3
            code = func.__code__
        except AttributeError:
            warnings.warn("Could not extract a code object for the object %r" % func)
        else:
            self.code_map.add(code)

    @contextmanager
    def _count_ctxmgr(self):
        self.enable_by_count()
        try:
            yield
        finally:
            self.disable_by_count()

    def wrap_function(self, func):
        """Wrap a function to profile it."""

        if iscoroutinefunction(func):

            @coroutine
            def f(*args, **kwargs):
                with self._count_ctxmgr():
                    res = yield from func(*args, **kwargs)
                    return res

        else:

            def f(*args, **kwds):
                with self._count_ctxmgr():
                    return func(*args, **kwds)

        return f

    def runctx(self, cmd, globals, locals):
        """Profile a single executable statement in the given namespaces."""
        self.enable_by_count()
        try:
            exec(cmd, globals, locals)
        finally:
            self.disable_by_count()
        return self

    def enable_by_count(self):
        """Enable the profiler if it hasn't been enabled before."""
        if self.enable_count == 0:
            self.enable()
        self.enable_count += 1

    def disable_by_count(self):
        """Disable the profiler if the number of disable requests matches the
        number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def trace_memory_usage(self, frame, event, arg):
        """Callback for sys.settrace"""
        if frame.f_code in self.code_map:
            if event == "call":
                # "call" event just saves the lineno but not the memory
                self.prevlines.append(frame.f_lineno)
            elif event == "line":
                # trace needs current line and previous line
                self.code_map.trace(frame.f_code, self.prevlines[-1], self.prev_lineno)
                # saving previous line
                self.prev_lineno = self.prevlines[-1]
                self.prevlines[-1] = frame.f_lineno
            elif event == "return":
                lineno = self.prevlines.pop()
                self.code_map.trace(frame.f_code, lineno, self.prev_lineno)
                self.prev_lineno = lineno

        if self._original_trace_function is not None:
            self._original_trace_function(frame, event, arg)

        return self.trace_memory_usage

    def trace_max_mem(self, frame, event, arg):
        # run into PDB as soon as memory is higher than MAX_MEM
        if event in ("line", "return") and frame.f_code in self.code_map:
            c = _get_memory(-1, self.backend, filename=frame.f_code.co_filename)
            if c >= self.max_mem:
                t = (
                    "Current memory {0:.2f} MiB exceeded the "
                    "maximum of {1:.2f} MiB\n".format(c, self.max_mem)
                )
                sys.stdout.write(t)
                sys.stdout.write("Stepping into the debugger \n")
                frame.f_lineno -= 2
                p = pdb.Pdb()
                p.quitting = False
                p.stopframe = frame
                p.returnframe = None
                p.stoplineno = frame.f_lineno - 3
                p.botframe = None
                return p.trace_dispatch

        if self._original_trace_function is not None:
            (self._original_trace_function)(frame, event, arg)

        return self.trace_max_mem

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()

    def enable(self):
        self._original_trace_function = sys.gettrace()
        if self.max_mem is not None:
            sys.settrace(self.trace_max_mem)
        else:
            sys.settrace(self.trace_memory_usage)

    def disable(self):
        sys.settrace(self._original_trace_function)


class LineProfiler(object):
    """ A profiler that records the amount of memory for each line """

    def __init__(self, **kw):
        include_children = kw.get("include_children", False)
        backend = kw.get("backend", "psutil")
        self.code_map = CodeMap(include_children=include_children, backend=backend)
        self.enable_count = 0
        self.max_mem = kw.get("max_mem", None)
        self.prevlines = []
        self.backend = choose_backend(kw.get("backend", None))
        self.prev_lineno = None

    def __call__(self, func=None, precision=1):
        if func is not None:
            self.add_function(func)
            f = self.wrap_function(func)
            f.__module__ = func.__module__
            f.__name__ = func.__name__
            f.__doc__ = func.__doc__
            f.__dict__.update(getattr(func, "__dict__", {}))
            return f
        else:

            def inner_partial(f):
                return self.__call__(f, precision=precision)

            return inner_partial

    def add_function(self, func):
        """Record line profiling information for the given Python function."""
        try:
            # func_code does not exist in Python3
            code = func.__code__
        except AttributeError:
            warnings.warn("Could not extract a code object for the object %r" % func)
        else:
            self.code_map.add(code)

    @contextmanager
    def _count_ctxmgr(self):
        self.enable_by_count()
        try:
            yield
        finally:
            self.disable_by_count()

    def wrap_function(self, func):
        """Wrap a function to profile it."""

        if iscoroutinefunction(func):

            @coroutine
            def f(*args, **kwargs):
                with self._count_ctxmgr():
                    res = yield from func(*args, **kwargs)
                    return res

        else:

            def f(*args, **kwds):
                with self._count_ctxmgr():
                    return func(*args, **kwds)

        return f

    def runctx(self, cmd, globals, locals):
        """Profile a single executable statement in the given namespaces."""
        self.enable_by_count()
        try:
            exec(cmd, globals, locals)
        finally:
            self.disable_by_count()
        return self

    def enable_by_count(self):
        """Enable the profiler if it hasn't been enabled before."""
        if self.enable_count == 0:
            self.enable()
        self.enable_count += 1

    def disable_by_count(self):
        """Disable the profiler if the number of disable requests matches the
        number of enable requests.
        """
        if self.enable_count > 0:
            self.enable_count -= 1
            if self.enable_count == 0:
                self.disable()

    def trace_memory_usage(self, frame, event, arg):
        """Callback for sys.settrace"""
        if frame.f_code in self.code_map:
            if event == "call":
                # "call" event just saves the lineno but not the memory
                self.prevlines.append(frame.f_lineno)
            elif event == "line":
                # trace needs current line and previous line
                self.code_map.trace(frame.f_code, self.prevlines[-1], self.prev_lineno)
                # saving previous line
                self.prev_lineno = self.prevlines[-1]
                self.prevlines[-1] = frame.f_lineno
            elif event == "return":
                lineno = self.prevlines.pop()
                self.code_map.trace(frame.f_code, lineno, self.prev_lineno)
                self.prev_lineno = lineno

        if self._original_trace_function is not None:
            self._original_trace_function(frame, event, arg)

        return self.trace_memory_usage

    def trace_max_mem(self, frame, event, arg):
        # run into PDB as soon as memory is higher than MAX_MEM
        if event in ("line", "return") and frame.f_code in self.code_map:
            c = _get_memory(-1, self.backend, filename=frame.f_code.co_filename)
            if c >= self.max_mem:
                t = (
                    "Current memory {0:.2f} MiB exceeded the "
                    "maximum of {1:.2f} MiB\n".format(c, self.max_mem)
                )
                sys.stdout.write(t)
                sys.stdout.write("Stepping into the debugger \n")
                frame.f_lineno -= 2
                p = pdb.Pdb()
                p.quitting = False
                p.stopframe = frame
                p.returnframe = None
                p.stoplineno = frame.f_lineno - 3
                p.botframe = None
                return p.trace_dispatch

        if self._original_trace_function is not None:
            (self._original_trace_function)(frame, event, arg)

        return self.trace_max_mem

    def __enter__(self):
        self.enable_by_count()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disable_by_count()

    def enable(self):
        self._original_trace_function = sys.gettrace()
        if self.max_mem is not None:
            sys.settrace(self.trace_max_mem)
        else:
            sys.settrace(self.trace_memory_usage)

    def disable(self):
        sys.settrace(self._original_trace_function)


def choose_backend(new_backend=None):
    """
    Function that tries to setup backend, chosen by user, and if failed,
    setup one of the allowable backends
    """

    _backend = "no_backend"
    all_backends = [
        ("psutil", True),
        ("psutil_pss", True),
        ("psutil_uss", True),
        ("posix", os.name == "posix"),
        ("tracemalloc", has_tracemalloc),
    ]
    backends_indices = dict((b[0], i) for i, b in enumerate(all_backends))

    if new_backend is not None:
        all_backends.insert(0, all_backends.pop(backends_indices[new_backend]))

    for n_backend, is_available in all_backends:
        if is_available:
            _backend = n_backend
            break
    if _backend != new_backend and new_backend is not None:
        warnings.warn(
            "{0} can not be used, {1} used instead".format(new_backend, _backend)
        )
    return _backend


def profile(
    func=None, memory=True, timer=True, stream=None, precision=1, backend="psutil"
):

    """
    Decorator that will run the function and print a line-by-line profile and total time
    """

    backend = choose_backend(backend)
    if backend == "tracemalloc" and has_tracemalloc:
        if not tracemalloc.is_tracing():
            tracemalloc.start()
    if func is not None:
        #        print(memory, timer)
        get_prof = partial(LineProfiler, backend=backend)
        show_results_bound = partial(show_results, stream=stream, precision=precision)
        if iscoroutinefunction(func):

            @wraps(wrapped=func)
            @coroutine
            def wrapper(*args, **kwargs):
                #                print(memory, timer)
                print(kwargs)
                prof = get_prof()
                tstart = time.time()
                val = yield from prof(func)(*args, **kwargs)
                tend = time.time()
                total_time = (tend - tstart) * 1000
                if timer:
                    print(
                        '"{}" took {:.3f} {} to execute\n'.format(
                            func.__name__,
                            total_time,
                            "ms",
                        )
                    )
                if memory:
                    show_results_bound(prof)
                return val

        else:

            @wraps(wrapped=func)
            def wrapper(*args, **kwargs):
                prof = get_prof()
                tstart = time.time()
                val = prof(func)(*args, **kwargs)
                tend = time.time()
                total_time = (tend - tstart) * 1000
                if timer:
                    print(
                        '"{}" took {:.3f} {} to execute\n'.format(
                            func.__name__,
                            total_time,
                            "ms",
                        )
                    )
                if memory:
                    show_results_bound(prof)
                return val

        return wrapper
    else:

        def inner_wrapper(f):
            return profile(
                f,
                stream=stream,
                precision=precision,
                backend=backend,
                memory=memory,
                timer=timer,
            )

        return inner_wrapper


if __name__ == "__main__":

    @profile(timer=True, memory=False)
    def delay(secs):
        import time

        time.sleep(secs)

    @profile
    def isPal(s):
        return s == s[::-1]

    print(isPal("OOO"))
    print(isPal("OOOSS"))
    delay(3)
