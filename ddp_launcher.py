import os
import sys

# Patch TCPStore to disable libuv on Windows before importing/running torchrun
if os.name == 'nt':
    try:
        import ctypes
        try:
            ctypes.windll.kernel32.SetEnvironmentVariableW('USE_LIBUV', '0')
        except Exception:
            pass
            
        try:
            ctypes.CDLL('msvcrt')._wputenv('USE_LIBUV=0')
        except Exception:
            pass
            
        import torch
        import torch.distributed
        import torch.distributed.rendezvous
        import torch.distributed.elastic.rendezvous.c10d_rendezvous_backend
        
        orig_tcp_store = torch.distributed.TCPStore
        
        def patched_tcp_store(*args, **kwargs):
            # Enforce use_libuv=False to bypass Windows libuv compilation issues
            kwargs['use_libuv'] = False
            return orig_tcp_store(*args, **kwargs)
            
        torch.distributed.TCPStore = patched_tcp_store
        
        # Dynamically patch sys.modules to propagate the patched TCPStore
        for mod_name, mod in list(sys.modules.items()):
            if mod is not None and hasattr(mod, '__dict__') and 'TCPStore' in mod.__dict__:
                if mod.__dict__['TCPStore'] is orig_tcp_store:
                    mod.__dict__['TCPStore'] = patched_tcp_store
    except Exception as e:
        print(f"Warning: Failed to apply TCPStore patch: {e}")

# Run standard torch.distributed.run main
from torch.distributed.run import main

if __name__ == '__main__':
    # Align sys.argv to look like standard torchrun call
    sys.argv[0] = '-m torch.distributed.run'
    main()
