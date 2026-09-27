"""Run a tiny CUDA workload and destroy its context to recover stuck GPU activity."""

import argparse
import ctypes as C
import os
import sys


# Use a private context on the explicitly selected device; no hardware reset or service shutdown.
def recover(bus_id):
    os.environ["CUDA_CACHE_DISABLE"] = "1"  # Keep the PTX compilation entirely in memory.
    cuda = C.CDLL("libcuda.so.1")

    # CUDA driver calls return status codes; report the failing API and native diagnostic.
    def call(name, argtypes, *args):
        function = getattr(cuda, name)
        function.argtypes = argtypes
        function.restype = C.c_int
        result = function(*args)
        if result:
            error = C.c_char_p()
            cuda.cuGetErrorString(C.c_int(result), C.byref(error))
            raise RuntimeError(f"{name}: {error.value.decode() if error.value else result}")

    context = C.c_void_p()
    module = C.c_void_p()
    pointer = C.c_uint64()
    try:
        call("cuInit", [C.c_uint], 0)
        device = C.c_int()
        call("cuDeviceGetByPCIBusId", [C.POINTER(C.c_int), C.c_char_p],
             C.byref(device), bus_id.encode("ascii"))
        print(f"Creating a temporary CUDA context on GPU {bus_id}", flush=True)
        call("cuCtxCreate_v2", [C.POINTER(C.c_void_p), C.c_uint, C.c_int],
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
        call("cuModuleLoadData", [C.POINTER(C.c_void_p), C.c_char_p], C.byref(module), ptx)
        kernel = C.c_void_p()
        call("cuModuleGetFunction", [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p],
             C.byref(kernel), module, b"recovery_probe")
        call("cuMemAlloc_v2", [C.POINTER(C.c_uint64), C.c_size_t], C.byref(pointer), 4)
        parameters = (C.c_void_p * 1)(C.cast(C.byref(pointer), C.c_void_p))
        print("Running and synchronizing a one-thread CUDA kernel", flush=True)
        call("cuLaunchKernel", [C.c_void_p] + [C.c_uint] * 7
             + [C.c_void_p, C.POINTER(C.c_void_p), C.POINTER(C.c_void_p)],
             kernel, 1, 1, 1, 1, 1, 1, 0, None, parameters, None)
        call("cuCtxSynchronize", [])
        result = C.c_uint32()
        call("cuMemcpyDtoH_v2", [C.c_void_p, C.c_uint64, C.c_size_t],
             C.byref(result), pointer, 4)
        if result.value != 42:
            raise RuntimeError(f"Unexpected GPU result: {result.value}")
        print("GPU computation verified", flush=True)
    finally:
        # Destroy the context even if allocation, execution, or individual resource cleanup fails.
        if context.value:
            try:
                if pointer.value:
                    call("cuMemFree_v2", [C.c_uint64], pointer)
                if module.value:
                    call("cuModuleUnload", [C.c_void_p], module)
            finally:
                print("Destroying the temporary CUDA context", flush=True)
                call("cuCtxDestroy_v2", [C.c_void_p], context)
                print("CUDA context shut down cleanly", flush=True)


# Require an explicit PCI address; CUDA ordinals need not match nvidia-smi indices.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pci_bus_id", help="PCI bus ID of the affected GPU, e.g. 0000:21:00.0 (Ampere or newer)")
    args = parser.parse_args()
    try:
        recover(args.pci_bus_id)
    except (OSError, RuntimeError, UnicodeError) as error:
        print(f"GPU recovery failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
