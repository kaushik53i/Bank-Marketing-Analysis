# =========================================================
# BANK MARKETING PROJECT - MAIN PIPELINE
# =========================================================

# =========================================================
# IMPORT LIBRARIES
# =========================================================

import subprocess
import time


# =========================================================
# FUNCTION TO RUN SCRIPTS
# =========================================================

def run_script(script_name):

    print("\n" + "=" * 60)

    print(f"🚀 RUNNING: {script_name}")

    print("=" * 60)

    start_time = time.time()

    result = subprocess.run(

        ["python", script_name],

        text=True
    )

    end_time = time.time()

    execution_time = end_time - start_time

    print("\n✅ COMPLETED:", script_name)

    print(f"⏱ Execution Time: {execution_time:.2f} seconds")


# =========================================================
# MAIN FUNCTION
# =========================================================

def main():

    print("\n" + "#" * 70)

    print("🏦 BANK MARKETING ANALYSIS PROJECT")

    print("#" * 70)

    print("""
Pipeline Steps:

1. SMOTE Preprocessing
2. Machine Learning Training
3. TOPSIS Ranking
4. Wilcoxon Statistical Testing
5. SHAP & LIME Explainability
""")


    # =====================================================
    # RUN ALL SCRIPTS
    # =====================================================

    run_script(
        "scripts/01_Smote_Preprocessing.py"
    )

    run_script(
        "scripts/02_Model_Training.py"
    )

    run_script(
        "scripts/03_TOPSIS_Ranking.py"
    )

    run_script(
        "scripts/04_Wilcoxon_Test.py"
    )

    run_script(
        "scripts/05_SHAP_LIME.py"
    )


    # =====================================================
    # FINAL MESSAGE
    # =====================================================

    print("\n" + "=" * 70)

    print("🎉 COMPLETE MACHINE LEARNING PIPELINE FINISHED")

    print("=" * 70)

    print("""
Generated Outputs:

✅ Balanced Dataset
✅ Model Evaluation Results
✅ TOPSIS Rankings
✅ Wilcoxon Statistical Results
✅ SHAP Graphs
✅ LIME Explanations
✅ Feature Importance Analysis
""")


# =========================================================
# RUN MAIN
# =========================================================

if __name__ == "__main__":

    main()