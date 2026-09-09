with open("axiomlm/kernels/metal_kernels.metal", "r") as f:
    content = f.read()

content = content.replace("    uint tid [[thread_position_in_threadgroup]],\n    uint2 bid [[threadgroup_position_in_grid]]",
                          "    uint2 tid_2d [[thread_position_in_threadgroup]],\n    uint2 bid [[threadgroup_position_in_grid]]")

content = content.replace("    uint q_idx = q_start + tid;", "    uint tid = tid_2d.x;\n    uint q_idx = q_start + tid;")

with open("axiomlm/kernels/metal_kernels.metal", "w") as f:
    f.write(content)
