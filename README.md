# audioset_raw

Download and create a tfreader for the audioset dataset

## Download dataset (takes few hours)

```bash
cd download/train
cat ../balanced_train_segments.csv | ../download.sh
cd ../validate 
cat ../eval_segments.csv | ../download.sh
```
## Build the TFRecor

```bash
cd download
python audioset_writer.py train
python audioset_writer.py validate
```
