import argparse
import os
import pandas as pd
from types import SimpleNamespace
from tools.inference_utils import FinCast_Inference

def main():
    parser = argparse.ArgumentParser(description="Run FinCast Inference")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", type=str, help="Comma-separated list of input values")
    group.add_argument("--data_path", type=str, help="Path to your input data (CSV)")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to your model (.pth) file")
    parser.add_argument("--save_output_path", type=str, required=False, help="Directory/Path to save your outputs", default=".")
    parser.add_argument("--backend", type=str, default="gpu", help="cpu or gpu")
    parser.add_argument("--data_frequency", type=str, default="m", help="Data frequency (s, m, h, d, w, y)")
    parser.add_argument("--context_len", type=int, default=128, help="Input length for forecast")
    parser.add_argument("--horizon_len", type=int, default=32, help="Output length for forecast")
    parser.add_argument("--columns_target", type=str, default="Close", help="Comma separated target column names")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    # Handle direct input data by creating a temporary CSV
    temp_csv_path = None
    if args.data:
        print("Parsing direct input data...")
        try:
            input_values = [float(x.strip()) for x in args.data.split(",")]
        except ValueError:
            print("Error: --data must be a comma-separated list of numbers.")
            return
            
        target_col = args.columns_target.split(",")[0]
        df = pd.DataFrame({target_col: input_values})
        temp_csv_path = "temp_input_data.csv"
        df.to_csv(temp_csv_path, index=False)
        data_path = temp_csv_path
    else:
        data_path = args.data_path

    config = SimpleNamespace()
    config.backend = args.backend
    config.model_path = args.model_path
    config.model_version = "v1"
    config.data_path = data_path
    config.data_frequency = args.data_frequency
    config.context_len = args.context_len
    config.horizon_len = args.horizon_len
    config.all_data = False
    config.columns_target = args.columns_target.split(",")
    config.series_norm = False
    config.batch_size = args.batch_size
    config.forecast_mode = "mean"
    config.quantile_outputs = []
    config.save_output = True
    config.save_output_path = args.save_output_path
    config.plt_outputs = False
    config.plt_quantiles = []

    print(f"Running FinCast Inference...")
    print(f"Model: {config.model_path}")
    
    fincast_inference = FinCast_Inference(config)
    preds, mapping, full_outputs = fincast_inference.run_inference()
    
    print(f"Inference complete!")
    print(f"Mean Predictions:\n{preds}")
    if args.save_output_path:
        print(f"Outputs saved to {config.save_output_path}")

    # Cleanup temporary CSV
    if temp_csv_path and os.path.exists(temp_csv_path):
        os.remove(temp_csv_path)

if __name__ == "__main__":
    main()
