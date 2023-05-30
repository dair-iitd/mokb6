for language in en hi te es pt zh
do
    baseline_data=./mokb6/mono/mono_${language}
    baseline_name=mono_${language}
    echo ${baseline_data}
    python convert_format_mokb.py --train ${baseline_data}/train.txt   --val ${baseline_data}/valid.txt --test ${baseline_data}/test.txt  --out_dir ./data/${baseline_name}

    python3 preprocess.py --train-path ./data/${baseline_name}/train.txt --valid-path ./data/${baseline_name}/valid.txt --test-path ./data/${baseline_name}/test.txt --task mopenkb
done