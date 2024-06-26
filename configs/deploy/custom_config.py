_base_ = ['./base_dynamic.py']
backend_config = dict(
    type='tensorrt',
    common_config=dict(fp16_mode=, max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[batch_size_choice, 3, 192, 192],
                    opt_shape=[batch_size_choice, 3, 640, 640],
                    max_shape=[batch_size_choice, 3, 960, 960])))
    ])
use_efficientnms = False  # whether to replace TRTBatchedNMS plugin with EfficientNMS plugin # noqa E501

