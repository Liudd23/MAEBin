#!/bin/bash

# Display help message if -h or --help is provided
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [FASTA_file] [OUTPUT_directory] [path_to_bamfiles]"
    echo
    echo "Arguments:"
    echo "  FASTA_file          Path to the input FASTA file"
    echo "  OUTPUT_directory    Path to the output directory"
    echo "  path_to_bamfiles    Path to the BAM files required for ComeBin"
    echo
    echo "Example:"
    echo "  $0 /path/to/input.fa /path/to/output /path/to/bamfiles"
    exit 0
fi

# Check if the required arguments are provided
if [[ -z "$1" || -z "$2" || -z "$3" ]]; then
    echo "Error: Missing arguments. Use -h or --help for usage information."
    exit 1
fi

# Assign input arguments to variables
contig_file=$1
output_path=$2
path_to_bamfiles=$3

# Step 1: Activate ComeBin environment and run ComeBin
echo "Activating ComeBin environment..."
conda activate ComeBin || { echo "Error: Failed to activate ComeBin environment."; exit 1; }

# Navigate to the ComeBin directory
cd COMEBin || { echo "Error: Failed to navigate to COMEBin directory."; exit 1; }

# Run ComeBin
CUDA_VISIBLE_DEVICES=0 bash run_comebin.sh -a ${contig_file} \
    -o ${output_path} \
    -p ${path_to_bamfiles} \
    -t 40 || { echo "Error: ComeBin execution failed."; exit 1; }

# Return to the previous directory
cd - || { echo "Error: Failed to return to the previous directory."; exit 1; }

# Step 2: Activate MetaCC environment and run MetaCC
echo "Activating MetaCC environment..."
conda activate MetaCC || { echo "Error: Failed to activate MetaCC environment."; exit 1; }

# Navigate to the MetaCC directory
cd MetaCC || { echo "Error: Failed to navigate to MetaCC directory."; exit 1; }

# Run MetaCC
python /path_to_MetaCC/MetaCC.py norm \
    [Parameters] ${contig_file} ${path_to_bamfiles} ${output_path} || { echo "Error: MetaCC execution failed."; exit 1; }

# Return to the previous directory
cd - || { echo "Error: Failed to return to the previous directory."; exit 1; }

# Step 3: Activate GraphMAE environment and run main_transductive.py
echo "Activating GraphMAE environment..."
conda activate GraphMAE || { echo "Error: Failed to activate GraphMAE environment."; exit 1; }

# Run main_transductive.py
dataset="Donor"
device=2

python main_transductive.py \
    --device $device \
    --dataset $dataset \
    --mask_rate 0.7 \
    --encoder "gat" \
    --decoder "gat" \
    --in_drop 0.2 \
    --attn_drop 0.1 \
    --num_layers 1 \
    --num_hidden 512 \
    --num_heads 2 \
    --max_epoch 1500 \
    --max_epoch_f 300 \
    --lr 0.001 \
    --weight_decay 0 \
    --lr_f 0.01 \
    --weight_decay_f 1e-4 \
    --activation prelu \
    --optimizer adam \
    --drop_edge_rate 0.0 \
    --loss_fn "sce" \
    --seeds 42 \
    --replace_rate 0.05 \
    --alpha_l 2 \
    --linear_prob \
    --scheduler || { echo "Error: main_transductive.py execution failed."; exit 1; }

# Completion message
echo "Pipeline completed successfully!"
