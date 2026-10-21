"""Private permissions for fresh test directories/files; never handles production keys."""

import os


def _set_acl(path, public: bool, directory: bool) -> None:
    import ctypes
    from ctypes import wintypes
    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi.OpenProcessToken.argtypes = [wintypes.HANDLE,wintypes.DWORD,ctypes.POINTER(wintypes.HANDLE)]
    advapi.GetTokenInformation.argtypes = [wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD)]
    advapi.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p,ctypes.POINTER(wintypes.LPWSTR)]
    advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(wintypes.DWORD)]
    advapi.SetFileSecurityW.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,ctypes.c_void_p]
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    token = wintypes.HANDLE()
    if not advapi.OpenProcessToken(kernel.GetCurrentProcess(),8,ctypes.byref(token)):
        raise RuntimeError("cannot query test owner")
    try:
        size = wintypes.DWORD()
        advapi.GetTokenInformation(token,1,None,0,ctypes.byref(size))
        data = ctypes.create_string_buffer(size.value)
        if not advapi.GetTokenInformation(token,1,data,size.value,ctypes.byref(size)):
            raise RuntimeError("cannot read test owner")
        sid = ctypes.cast(data,ctypes.POINTER(ctypes.c_void_p))[0]
        text = wintypes.LPWSTR()
        if not advapi.ConvertSidToStringSidW(sid,ctypes.byref(text)):
            raise RuntimeError("cannot encode test owner")
        try:
            inherit = "OICI" if directory else ""
            sddl = f"D:P(A;{inherit};FA;;;SY)(A;{inherit};FA;;;BA)(A;{inherit};FA;;;{text.value})"
            if public:
                sddl += f"(A;{inherit};GR;;;WD)"
        finally:
            kernel.LocalFree(ctypes.cast(text,ctypes.c_void_p))
        descriptor = ctypes.c_void_p()
        if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl,1,ctypes.byref(descriptor),None):
            raise RuntimeError("cannot construct test access policy")
        try:
            if not advapi.SetFileSecurityW(str(path),0x80000004,descriptor):
                raise RuntimeError("cannot set test access policy")
        finally:
            kernel.LocalFree(descriptor)
    finally:
        kernel.CloseHandle(token)


def secure_directory(path) -> None:
    if os.name == "nt":
        _set_acl(path,False,True)
    else:
        os.chmod(path,0o700)


def make_public_test_file(path) -> None:
    if os.name == "nt":
        _set_acl(path,True,False)
    else:
        os.chmod(path,0o644)
