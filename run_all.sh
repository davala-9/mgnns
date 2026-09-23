for config in ./configs/*.yaml; do
    echo "Running with $config"
    python -m src.run.run_experiment "$config"
done
