#!/bin/bash
# Example script to run multi-scale evaluation
# This demonstrates basic usage of the evaluation tool

echo "================================================"
echo "Multi-Scale Evaluation - Example Run"
echo "================================================"
echo ""

# Step 1: List available scenes
echo "Step 1: Listing available scenes..."
echo "------------------------------------------------"
python list_scenes.py | head -30
echo ""
echo "See full list with: python list_scenes.py"
echo ""

# Step 2: Run a quick test evaluation
SCENE="0000"
NUM_PAIRS=10
OUTPUT_DIR="./example_results"

echo "Step 2: Running evaluation on scene $SCENE"
echo "------------------------------------------------"
echo "Configuration:"
echo "  - Scene: $SCENE"
echo "  - Number of pairs: $NUM_PAIRS"
echo "  - Output directory: $OUTPUT_DIR"
echo ""

python multi_scale_eval.py \
    --scene $SCENE \
    --num_pairs $NUM_PAIRS \
    --output_dir $OUTPUT_DIR

# Step 3: Show results
echo ""
echo "Step 3: Results"
echo "------------------------------------------------"

if [ -f "${OUTPUT_DIR}/${SCENE}_results.csv" ]; then
    echo "✓ CSV results saved to: ${OUTPUT_DIR}/${SCENE}_results.csv"
    echo ""
    echo "First few rows:"
    head -5 "${OUTPUT_DIR}/${SCENE}_results.csv" | column -t -s,
fi

if [ -f "${OUTPUT_DIR}/${SCENE}_distance_vs_accuracy.png" ]; then
    echo ""
    echo "✓ Plot saved to: ${OUTPUT_DIR}/${SCENE}_distance_vs_accuracy.png"
fi

echo ""
echo "================================================"
echo "Example run complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "  1. View the CSV: cat ${OUTPUT_DIR}/${SCENE}_results.csv"
echo "  2. View the plot: Open ${OUTPUT_DIR}/${SCENE}_distance_vs_accuracy.png"
echo "  3. Run with more pairs: python multi_scale_eval.py --scene $SCENE --num_pairs 30"
echo "  4. Try a different scene: python multi_scale_eval.py --scene 0001"
echo ""

