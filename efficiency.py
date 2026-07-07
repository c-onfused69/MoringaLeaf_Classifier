import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from config import Config
from models.architectures import build_model

def get_flops(model):
    """
    Estimates the FLOPs (Floating Point Operations) of a tf.keras Model.
    Uses the tf.compat.v1.profiler.
    """
    try:
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
        
        # Define a tf.function to trace the model
        @tf.function
        def model_func(inputs):
            return model(inputs)
            
        # Get concrete function
        input_signature = [tf.TensorSpec(shape=(1,) + model.input_shape[1:], dtype=tf.float32)]
        concrete_func = model_func.get_concrete_function(*input_signature)
        
        # Convert to frozen graph
        frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
        
        # Run profiler
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            # Suppress excessive output
            opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd='op', options=opts)
            
            return flops.total_float_ops if flops else 0
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs. Reason: {e}")
        return 0

def measure_efficiency(model_name, num_iterations=100):
    """
    Measures parameters, size, and CPU inference latency for a given model.
    """
    input_shape = Config.INPUT_SHAPE_INCEPTION if model_name.lower() == "inception_v3" else Config.INPUT_SHAPE_DEFAULT
    
    # Build model (untrained, we just need the architecture)
    model, _, _ = build_model(model_name, input_shape, Config.NUM_CLASSES)
    
    # 1. Total Parameters
    total_params = model.count_params()
    
    # 2. Model Size Estimate (in MB)
    # Assuming float32 (4 bytes per parameter)
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    # 3. Estimated FLOPs
    flops = get_flops(model)
    gflops = flops / 1e9 if flops else 0
    
    # 4. Inference Latency (CPU)
    # Force CPU for edge deployment simulation
    dummy_input = np.random.random((1, *input_shape)).astype(np.float32)
    
    with tf.device('/CPU:0'):
        # Warmup (first prediction is always slow due to tracing/initialization)
        for _ in range(10):
            _ = model.predict(dummy_input, verbose=0)
            
        # Measure latency
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model.predict(dummy_input, verbose=0)
        end_time = time.time()
        
    latency_ms = ((end_time - start_time) / num_iterations) * 1000
    
    return {
        "Model": model_name,
        "Total_Parameters (M)": round(total_params / 1e6, 2),
        "Model_Size (MB)": round(model_size_mb, 2),
        "Estimated_GFLOPs": round(gflops, 3) if gflops > 0 else "N/A",
        "CPU_Latency (ms)": round(latency_ms, 2)
    }

def main():
    print("Running Efficiency Profiling on CPU (simulating edge deployment)...")
    
    # Define hardware specs based on user configuration
    hardware_info = "CPU i7-1165G7, GPU: Intel Iris(R) Xe Graphics, RAM: 16GB"
    print(f"Hardware Profile: {hardware_info}\n")
    
    models_to_test = ["inception_v3", "resnet50", "densenet121", "efficientnetb0", "mobilenetv3", "vit"]
    results = []
    
    for model_name in models_to_test:
        print(f"Profiling {model_name}...")
        try:
            metrics = measure_efficiency(model_name)
            results.append(metrics)
        except Exception as e:
            print(f"Failed to profile {model_name}: {e}")
            
    # Save to CSV
    df = pd.DataFrame(results)
    out_dir = os.path.join(Config.OUTPUTS_DIR, "efficiency")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "model_efficiency_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    print("\n--- Profiling Complete ---")
    print(df.to_string(index=False))
    print(f"\nResults saved to {csv_path}")

if __name__ == "__main__":
    main()
