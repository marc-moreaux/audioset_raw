# audioset_raw

Download and create a tfreader for the audioset dataset

## 1) Download dataset (takes few hours)

```bash
cd download/train
cat ../balanced_train_segments.csv | ../download.sh
cd ../validate 
cat ../eval_segments.csv | ../download.sh
```

## 2-a) Build the TFRecord file

```bash
cd download
python audioset_writer.py train
python audioset_writer.py validate
```

## 2-b) Access WAV through a simple generator
``` python
from audioset_generator import get_batchs

# get_batchs is a generator
my_gen = get_batchs(batch_size=10, n_epochs=1 num_threads=4)
my_gen.next()
```
