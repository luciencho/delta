# jddc_belta

### For Data Genration
```
python datagen.py
    --data_dir $DATA_DIR
    --tmp_dir $TMP_DIR
    --hparams $HPARAMS
```
### For Neural Network Retrieval Model Training
```
python trainer.py
    --tmp_dir $TMP_DIR
    --model_dir $MODEL_DIR
    --hparams $HPARAMS
    --problems retrieval
python vector.py
    --tmp_dir $TMP_DIR
    --model_dir $MODEL_DIR
    --hparams $HPARAMS
```
### For Traditional Retrieval Model Training
```
python trainer.py
    --tmp_dir $TMP_DIR
    --model_dir $MODEL_DIR
    --hparams $HPARAMS
    --problems keyword
python trainer.py
    --tmp_dir $TMP_DIR
    --model_dir $MODEL_DIR
    --hparams $HPARAMS
    --problems retrieval_trad

```
### For File-to-File Decoding
```
python searcher.py input_file output_file
```