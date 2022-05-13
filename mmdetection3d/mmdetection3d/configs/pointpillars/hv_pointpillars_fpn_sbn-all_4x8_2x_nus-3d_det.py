_base_ = [
    '../_base_/models/hv_pointpillars_fpn_nus_basic.py',
    '../_base_/datasets/nus-3d_basic.py', '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
evaluation = dict(interval=2)
find_unused_parameters = True
