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
    --gpu_device 0
    --gpu_memory 0.23
    --problems dual_encoder
```
### For TFIDF Retrieval Model Training
```
python trainer.py
    --tmp_dir $TMP_DIR
    --model_dir $MODEL_DIR
    --hparams $HPARAMS
    --problems tfidf
```
### For LDA Retrieval Model Training
```
python trainer.py
    --tmp_dir $TMP_DIR
    --model_dir $MODEL_DIR
    --hparams $HPARAMS
    --problems lda

```
### For File-to-File Decoding
```
python searcher.py input_file output_file
```