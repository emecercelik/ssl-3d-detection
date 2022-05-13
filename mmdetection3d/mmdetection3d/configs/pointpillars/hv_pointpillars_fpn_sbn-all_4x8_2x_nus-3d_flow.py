_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus_flow.py',
    '../_base_/datasets/nus-3d_flow.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
find_unused_parameters = True