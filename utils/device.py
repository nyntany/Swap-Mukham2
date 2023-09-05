import onnx
import onnxruntime
import multiprocessing as mp

device_types_list = ["cpu", "cuda"]

available_providers = onnxruntime.get_available_providers()

def get_device_and_provider(device='cpu'):
    options = onnxruntime.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    use_num_cpus = mp.cpu_count()-1
    
    if device == 'cuda':
        if "CUDAExecutionProvider" in available_providers:
            provider = [
            ('CUDAExecutionProvider', {
                'tunable_op_enable': 1, 
                'tunable_op_tuning_enable': 1,
                'cudnn_conv1d_pad_to_nc1d': 0,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            })
        ]
            options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            options.intra_op_num_threads = 2
        else:
            device = 'cpu'
            provider = ["CPUExecutionProvider"]
    else:
        device = 'cpu'
        provider = ["CPUExecutionProvider"]

    return device, provider, options


data_type_bytes = {'uint8': 1, 'int8': 1, 'uint16': 2, 'int16': 2, 'float16': 2, 'float32': 4}


def estimate_max_batch_size(resolution, chunk_size=1024, data_type='float32', channels=3):
    pixel_size = data_type_bytes.get(data_type, 1)
    image_size = resolution[0] * resolution[1] * pixel_size * channels
    number_of_batches = (chunk_size * 1024 * 1024) // image_size
    return max(number_of_batches, 1)
