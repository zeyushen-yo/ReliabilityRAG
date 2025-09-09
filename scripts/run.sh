#!/bin/bash
declare -a defenses=( "sampleMIS" )
declare -a datasets=( "realtimeqa_sorted" )
declare -a models=( "mistral7b" )
declare -a attacks=( "PIA" )
declare -a attackpositions=( 0 24 49 )
declare -a corruption_size=( 1 )
declare -a gammas=( 0.9 )
declare -a errs=( 0.1 0.3 0.5 )
export AI_SANDBOX_KEY="ae696dffa9874e17bbd7f13499d3b571"
export OPENAI_API_KEY="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"
for defense in "${defenses[@]}"; do
    for data in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            for attack in "${attacks[@]}"; do
                for attackpos in "${attackpositions[@]}"; do
                    for corruption_size in "${corruption_size[@]}"; do
                        for gamma in "${gammas[@]}"; do
                            for err in "${errs[@]}"; do
                                echo "executing $model-$data-$defense-$attack-$attackpos-$corruption_size-$gamma-$err"
                                sbatch scripts/run.slurm $defense $data $model $attack $attackpos $corruption_size $gamma $err
                                # python main.py --model_name $model --dataset_name $data --top_k 50 --defense_method $defense --gamma $gamma --attack_method $attack --attackpos $attackpos --rep 1 --m 2 --T 20 --debug --model_dir="/scratch/gpfs/zs7353/" --save_response
                            done
                        done
                    done
                done
            done
        done
    done
done