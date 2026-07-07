import os
import pandas as pd
from config import Config

def df_to_latex(df, caption, label):
    """
    Converts a pandas DataFrame to a publication-ready LaTeX table.
    """
    latex_str = df.to_latex(index=False, float_format="%.3f")
    
    # Wrap in table environment with formatting for IEEE/MDPI
    table_env = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        latex_str.strip(),
        "}",
        "\\end{table}"
    ]
    return "\n".join(table_env)

def generate_efficiency_table(out_dir):
    csv_path = os.path.join(Config.OUTPUTS_DIR, "efficiency", "model_efficiency_metrics.csv")
    if not os.path.exists(csv_path):
        print(f"Efficiency metrics not found at {csv_path}. Run efficiency.py first.")
        return
        
    df = pd.read_csv(csv_path)
    
    latex_code = df_to_latex(
        df, 
        "Computational complexity and inference latency of the evaluated deep learning models on Intel Core i7-1165G7 CPU.", 
        "tab:efficiency"
    )
    
    with open(os.path.join(out_dir, "table_efficiency.tex"), "w") as f:
        f.write(latex_code)
    print("Exported table_efficiency.tex")

def generate_ablation_table(out_dir):
    ablation_dir = os.path.join(Config.OUTPUTS_DIR, "ablation")
    if not os.path.exists(ablation_dir):
        print(f"Ablation metrics not found. Run ablation.py first.")
        return
        
    # Combine all ablation csvs
    all_dfs = []
    for f in os.listdir(ablation_dir):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(ablation_dir, f))
            all_dfs.append(df)
            
    if not all_dfs:
        print("No ablation CSVs found.")
        return
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    latex_code = df_to_latex(
        combined_df, 
        "Ablation study results detailing the impact of data augmentation and fine-tuning strategies on model performance.", 
        "tab:ablation"
    )
    
    with open(os.path.join(out_dir, "table_ablation.tex"), "w") as f:
        f.write(latex_code)
    print("Exported table_ablation.tex")

def main():
    print("Generating publication tables...")
    
    tables_dir = os.path.join("paper", "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    generate_efficiency_table(tables_dir)
    generate_ablation_table(tables_dir)
    
    print(f"Done! Check the {tables_dir} directory for your LaTeX files.")

if __name__ == "__main__":
    main()
