"""List GPUs with their utilization, or recover one GPU stuck busy with no process.

Without an argument, print every GPU as nvidia-smi numbers it, with utilization,
memory in use, and running compute processes: a GPU at full utilization with no
process is the condition this tool clears. With a GPU number, run a tiny CUDA
kernel on that GPU and destroy its context, which releases activity a killed
process left behind. No hardware reset, no service shutdown.
"""

import argparse
import contextlib
import ctypes as C
import os
import sys

NVML_ERROR_INSUFFICIENT_SIZE = 7


# nvmlUtilization_t: percentages over NVML's last sample period.
class _Utilization(C.Structure):
    _fields_ = [("gpu", C.c_uint), ("memory", C.c_uint)]


# nvmlMemory_t (v1), bytes.
class _Memory(C.Structure):
    _fields_ = [("total", C.c_ulonglong), ("free", C.c_ulonglong), ("used", C.c_ulonglong)]


# nvmlPciInfo_t (v3). bus_id_legacy is the "0000:21:00.0" form the CUDA driver's
# cuDeviceGetByPCIBusId documents; it bridges NVML's numbering to CUDA's.
class _PciInfo(C.Structure):
    _fields_ = [("bus_id_legacy", C.c_char * 16), ("domain", C.c_uint), ("bus", C.c_uint),
                ("device", C.c_uint), ("pci_device_id", C.c_uint), ("pci_subsystem_id", C.c_uint),
                ("bus_id", C.c_char * 32)]


# nvmlProcessInfo_t as nvmlDeviceGetComputeRunningProcesses_v3 fills it.
class _ProcessInfo(C.Structure):
    _fields_ = [("pid", C.c_uint), ("used_gpu_memory", C.c_ulonglong),
                ("gpu_instance_id", C.c_uint), ("compute_instance_id", C.c_uint)]


# Load a driver library and wrap its status-code convention: every call names the
# failing API and carries the library's own message. `accept` lists nonzero codes
# the caller treats as answers rather than failures; the code is returned.
def _load(path, describe):
    lib = C.CDLL(path)

    def call(name, argtypes, *args, accept=()):
        function = getattr(lib, name)
        function.argtypes = argtypes
        function.restype = C.c_int
        result = function(*args)
        if result and result not in accept:
            raise RuntimeError(f"{name}: {describe(lib, result)}")
        return result

    return call


def _cuda_error(lib, code):
    error = C.c_char_p()
    lib.cuGetErrorString(C.c_int(code), C.byref(error))
    return error.value.decode() if error.value else str(code)


def _nvml_error(lib, code):
    lib.nvmlErrorString.argtypes = [C.c_int]
    lib.nvmlErrorString.restype = C.c_char_p
    message = lib.nvmlErrorString(code)
    return message.decode() if message else str(code)


# NVML numbers GPUs the way nvidia-smi and monitoring tools do, independent of
# CUDA_VISIBLE_DEVICES; every operator-facing number in this tool is an NVML index.
@contextlib.contextmanager
def _nvml():
    nvml = _load("libnvidia-ml.so.1", _nvml_error)
    nvml("nvmlInit_v2", [])
    try:
        yield nvml
    finally:
        nvml("nvmlShutdown", [])


def _device_count(nvml):
    count = C.c_uint()
    nvml("nvmlDeviceGetCount_v2", [C.POINTER(C.c_uint)], C.byref(count))
    return count.value


# Reject a bad number here so the message names the valid range instead of NVML's
# bare "Invalid Argument".
def _handle(nvml, index):
    count = _device_count(nvml)
    if not 0 <= index < count:
        raise RuntimeError(f"GPU {index} does not exist; this host has GPUs 0 to {count - 1}")
    handle = C.c_void_p()
    nvml("nvmlDeviceGetHandleByIndex_v2", [C.c_uint, C.POINTER(C.c_void_p)], index, C.byref(handle))
    return handle


# Asked with no buffer, NVML answers insufficient-size with a count that may be
# padded against races, so a second call with a buffer gives the exact number.
def _process_count(nvml, handle):
    argtypes = [C.c_void_p, C.POINTER(C.c_uint), C.POINTER(_ProcessInfo)]
    needed = C.c_uint(0)
    status = nvml("nvmlDeviceGetComputeRunningProcesses_v3", argtypes, handle, C.byref(needed), None,
                  accept=(NVML_ERROR_INSUFFICIENT_SIZE,))
    if status == 0:
        return 0
    buffer = (_ProcessInfo * (needed.value * 2 + 5))()
    filled = C.c_uint(len(buffer))
    nvml("nvmlDeviceGetComputeRunningProcesses_v3", argtypes, handle, C.byref(filled), buffer)
    return filled.value


def _row(nvml, index):
    handle = _handle(nvml, index)
    name = C.create_string_buffer(96)
    nvml("nvmlDeviceGetName", [C.c_void_p, C.c_char_p, C.c_uint], handle, name, len(name))
    utilization = _Utilization()
    nvml("nvmlDeviceGetUtilizationRates", [C.c_void_p, C.POINTER(_Utilization)], handle, C.byref(utilization))
    memory = _Memory()
    nvml("nvmlDeviceGetMemoryInfo", [C.c_void_p, C.POINTER(_Memory)], handle, C.byref(memory))
    return index, name.value.decode(), utilization.gpu, memory.used, memory.total, _process_count(nvml, handle)


def _bus_id(nvml, index):
    pci = _PciInfo()
    nvml("nvmlDeviceGetPciInfo_v3", [C.c_void_p, C.POINTER(_PciInfo)], _handle(nvml, index), C.byref(pci))
    return pci.bus_id_legacy.decode()


# Print the table an operator needs to pick a GPU: the stuck one shows high
# utilization with zero processes.
def list_gpus():
    with _nvml() as nvml:
        rows = [_row(nvml, index) for index in range(_device_count(nvml))]
    mib = 1024 * 1024
    lines = [("GPU", "Utilization", "Memory used", "Processes", "Name")]
    for index, name, utilization, used, total, processes in rows:
        lines.append((str(index), f"{utilization}%", f"{used // mib} / {total // mib} MiB", str(processes), name))
    widths = [max(len(line[column]) for line in lines) for column in range(4)]
    for line in lines:
        print("  ".join(cell.ljust(width) for cell, width in zip(line[:4], widths)) + "  " + line[4])
    print("A GPU at high utilization with 0 processes is stuck; run `recover-gpu <GPU>` to clear it.")


# Use a private context on the selected device; no hardware reset or service shutdown.
def recover(index):
    with _nvml() as nvml:
        bus_id = _bus_id(nvml, index)
    os.environ["CUDA_CACHE_DISABLE"] = "1"  # Keep the PTX compilation entirely in memory.
    cuda = _load("libcuda.so.1", _cuda_error)

    context = C.c_void_p()
    module = C.c_void_p()
    pointer = C.c_uint64()
    try:
        cuda("cuInit", [C.c_uint], 0)
        device = C.c_int()
        cuda("cuDeviceGetByPCIBusId", [C.POINTER(C.c_int), C.c_char_p],
             C.byref(device), bus_id.encode("ascii"))
        print(f"Creating a temporary CUDA context on GPU {index} ({bus_id})", flush=True)
        cuda("cuCtxCreate_v2", [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
             C.byref(context), 0, device)
        # Execute an SM kernel, then verify its output rather than just allocating memory.
        ptx = b""".version 7.0
.target sm_80
.address_size 64
.visible .entry recovery_probe(.param .u64 output) {
    .reg .u64 %address;
    .reg .u32 %value;
    ld.param.u64 %address, [output];
    mov.u32 %value, 42;
    st.global.u32 [%address], %value;
    ret;
}
"""
        cuda("cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_char_p], C.byref(module), ptx)
        kernel = C.c_void_p()
        cuda("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
             C.byref(kernel), module, b"recovery_probe")
        cuda("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t], C.byref(pointer), 4)
        parameters = (C.c_void_p * 1)(C.cast(C.byref(pointer), C.c_void_p))
        print("Running and synchronizing a one-thread CUDA kernel", flush=True)
        cuda("cuLaunchKernel", [C.c_void_p] + [C.c_uint] * 7
             + [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)],
             kernel, 1, 1, 1, 1, 1, 1, 0, None, parameters, None)
        cuda("cuCtxSynchronize", [])
        result = C.c_uint32()
        cuda("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t],
             C.byref(result), pointer, 4)
        if result.value != 42:
            raise RuntimeError(f"Unexpected GPU result: {result.value}")
        print("GPU computation verified", flush=True)
    finally:
        # Destroy the context even if allocation, execution, or individual resource cleanup fails.
        if context.value:
            try:
                if pointer.value:
                    cuda("cuMemFree_v2", [C.c_uint64], pointer)
                if module.value:
                    cuda("cuModuleUnload", [C.c_void_p], module)
            finally:
                print("Destroying the temporary CUDA context", flush=True)
                cuda("cuCtxDestroy_v2", [C.c_void_p], context)
                print("CUDA context shut down cleanly", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gpu", nargs="?", type=int,
                        help="GPU number as listed by this tool, nvidia-smi, or btop; omit to list GPUs")
    args = parser.parse_args()
    try:
        if args.gpu is None:
            list_gpus()
        else:
            recover(args.gpu)
    except (OSError, RuntimeError, UnicodeError) as error:
        print(f"recover-gpu: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
