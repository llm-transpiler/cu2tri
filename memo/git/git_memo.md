git submodule update --remote third_party/xpiler-eval

cd /workspace/third_party/
git submodule add https://github.com/apache/tvm tvm
cd tvm
git submodule update --init --recursive
git checkout v0.22.dev0
git submodule update --init --recursive

# tvm record
root@ubuntu-ThinkStation-P520:/workspace/third_party# git submodule add https://github.com/apache/tvm tvm
Cloning into '/workspace/third_party/tvm'...
remote: Enumerating objects: 228265, done.
remote: Counting objects: 100% (116/116), done.
remote: Compressing objects: 100% (90/90), done.
remote: Total 228265 (delta 54), reused 30 (delta 26), pack-reused 228149 (from 2)
Receiving objects: 100% (228265/228265), 114.72 MiB | 935.00 KiB/s, done.
Resolving deltas: 100% (176493/176493), done.
root@ubuntu-ThinkStation-P520:/workspace/third_party# cd tvm
root@ubuntu-ThinkStation-P520:/workspace/third_party/tvm# git submodule update --init --recursive
Submodule '3rdparty/OpenCL-Headers' (https://github.com/KhronosGroup/OpenCL-Headers.git) registered for path '3rdparty/OpenCL-Headers'
Submodule '3rdparty/cnpy' (https://github.com/rogersce/cnpy.git) registered for path '3rdparty/cnpy'
Submodule '3rdparty/cutlass' (https://github.com/NVIDIA/cutlass.git) registered for path '3rdparty/cutlass'
Submodule '3rdparty/cutlass_fpA_intB_gemm' (https://github.com/tlc-pack/cutlass_fpA_intB_gemm) registered for path '3rdparty/cutlass_fpA_intB_gemm'
Submodule 'dmlc-core' (https://github.com/dmlc/dmlc-core.git) registered for path '3rdparty/dmlc-core'
Submodule '3rdparty/libflash_attn' (https://github.com/tlc-pack/libflash_attn) registered for path '3rdparty/libflash_attn'
Submodule '3rdparty/rang' (https://github.com/agauniyal/rang.git) registered for path '3rdparty/rang'
Submodule '3rdparty/tvm-ffi' (https://github.com/apache/tvm-ffi) registered for path '3rdparty/tvm-ffi'
Submodule '3rdparty/zlib' (https://github.com/madler/zlib.git) registered for path '3rdparty/zlib'
Cloning into '/workspace/third_party/tvm/3rdparty/OpenCL-Headers'...
Cloning into '/workspace/third_party/tvm/3rdparty/cnpy'...
Cloning into '/workspace/third_party/tvm/3rdparty/cutlass'...
Cloning into '/workspace/third_party/tvm/3rdparty/cutlass_fpA_intB_gemm'...
Cloning into '/workspace/third_party/tvm/3rdparty/dmlc-core'...
Cloning into '/workspace/third_party/tvm/3rdparty/libflash_attn'...
Cloning into '/workspace/third_party/tvm/3rdparty/rang'...
Cloning into '/workspace/third_party/tvm/3rdparty/tvm-ffi'...
Cloning into '/workspace/third_party/tvm/3rdparty/zlib'...
Submodule path '3rdparty/OpenCL-Headers': checked out 'b590a6bfe034ea3a418b7b523e3490956bcb367a'
Submodule path '3rdparty/cnpy': checked out '4e8810b1a8637695171ed346ce68f6984e585ef4'
Submodule path '3rdparty/cutlass': checked out 'b2dd65dc864e09688245b316ac46c4a6cd07e15c'
Submodule path '3rdparty/cutlass_fpA_intB_gemm': checked out '72b9883c986a2ff427ca61ac0b14ad59be1dc862'
Submodule 'cutlass' (https://github.com/NVIDIA/cutlass) registered for path '3rdparty/cutlass_fpA_intB_gemm/cutlass'
Cloning into '/workspace/third_party/tvm/3rdparty/cutlass_fpA_intB_gemm/cutlass'...
git checkout v0.22.dev0
Submodule path '3rdparty/cutlass_fpA_intB_gemm/cutlass': checked out 'cc85b64cf676c45f98a17e3a47c0aafcf817f088'
Submodule path '3rdparty/dmlc-core': checked out '3031e4a61a98f49f07a42cfdec6242340fb2fd8c'
Submodule path '3rdparty/libflash_attn': checked out '07ba35bae96900e51b4d63ef3487ba24850da870'
Submodule 'cutlass' (https://github.com/NVIDIA/cutlass/) registered for path '3rdparty/libflash_attn/cutlass'
Cloning into '/workspace/third_party/tvm/3rdparty/libflash_attn/cutlass'...
Submodule path '3rdparty/libflash_attn/cutlass': checked out 'e0aaa3c3b38db9a89c31f04fef91e92123ad5e2e'
Submodule path '3rdparty/rang': checked out 'cabe04d6d6b05356fa8f9741704924788f0dd762'
Submodule path '3rdparty/tvm-ffi': checked out '4fefeb0f5913fc41cf860f517b9320f1bf1d0e98'
Submodule '3rdparty/dlpack' (https://github.com/dmlc/dlpack) registered for path '3rdparty/tvm-ffi/3rdparty/dlpack'
Submodule '3rdparty/libbacktrace' (https://github.com/ianlancetaylor/libbacktrace) registered for path '3rdparty/tvm-ffi/3rdparty/libbacktrace'
Cloning into '/workspace/third_party/tvm/3rdparty/tvm-ffi/3rdparty/dlpack'...
Cloning into '/workspace/third_party/tvm/3rdparty/tvm-ffi/3rdparty/libbacktrace'...
Submodule path '3rdparty/tvm-ffi/3rdparty/dlpack': checked out 'addbc8b3d9449691d01827ac4a0e0d035cf8ea40'
Submodule path '3rdparty/tvm-ffi/3rdparty/libbacktrace': checked out '793921876c981ce49759114d7bb89bb89b2d3a2d'
Submodule path '3rdparty/zlib': checked out 'ef24c4c7502169f016dcd2a26923dbaf3216748c'
root@ubuntu-ThinkStation-P520:/workspace/third_party/tvm# git checkout v0.22.dev0
warning: unable to rmdir '3rdparty/tvm-ffi': Directory not empty
Updating files: 100% (1801/1801), done.
M       3rdparty/cutlass
M       3rdparty/cutlass_fpA_intB_gemm
Note: switching to 'v0.22.dev0'.

You are in 'detached HEAD' state. You can look around, make experimental
changes and commit them, and you can discard any commits you make in this
state without impacting any branches by switching back to a branch.

If you want to create a new branch to retain commits you create, you may
do so (now or later) by using -c with the switch command. Example:

  git switch -c <new-branch-name>

Or undo this operation with:

  git switch -

Turn off this advice by setting config variable advice.detachedHead to false

HEAD is now at 045eb5bc9 [release] Update version to 0.22.dev0 on main branch
root@ubuntu-ThinkStation-P520:/workspace/third_party/tvm# git checkout v0.22.dev0^C
root@ubuntu-ThinkStation-P520:/workspace/third_party/tvm# # 确保你仍在 tvm 目录下
git submodule update --init --recursive
Submodule 'dlpack' (https://github.com/dmlc/dlpack.git) registered for path '3rdparty/dlpack'
Submodule '3rdparty/libbacktrace' (https://github.com/tlc-pack/libbacktrace.git) registered for path '3rdparty/libbacktrace'
Submodule 'ffi/3rdparty/dlpack' (https://github.com/dmlc/dlpack.git) registered for path 'ffi/3rdparty/dlpack'
Cloning into '/workspace/third_party/tvm/3rdparty/dlpack'...
Cloning into '/workspace/third_party/tvm/3rdparty/libbacktrace'...
Cloning into '/workspace/third_party/tvm/ffi/3rdparty/dlpack'...
Submodule path '3rdparty/cutlass': checked out 'ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e'
Submodule path '3rdparty/cutlass_fpA_intB_gemm': checked out 'b71d94a4ccd6573c9cbd4056c9ce660f110d33f0'
Submodule path '3rdparty/dlpack': checked out '3ea601bb413074c49a77c4ce3218bc08f8c4703c'
Submodule path '3rdparty/libbacktrace': checked out '08f7c7e69f8ea61a0c4151359bc8023be8e9217b'
Submodule path 'ffi/3rdparty/dlpack': checked out '3ea601bb413074c49a77c4ce3218bc08f8c4703c'


