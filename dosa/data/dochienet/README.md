### Utilities to process, train and evaluate DocHieNet dataset
Follow instruction below to download DocHieNet Dataset or refer [DocHieNet DataSet](https://modelscope.cn/datasets/iic/DocHieNet).
```bash
pip install modelscope
mkdir dochienet
modelscope download --dataset iic/DocHieNet --local_dir dochienet
cd dochienet
cat dochienet_dataset.zip.part-* > dochienet_dataset.zip
unzip dochienet_dataset.zip
cd dochienet_dataset
```

After the dataset is download, run process.py to process data into dosa format.
```bash
python dosa/data/dochienet/preprocess.py --in_path ./input_dir --out_path ./output_dir --resolution 1024 1024 --limit 512 --window 16 --skips doc1 doc2
```

Basically, the preprocess.py has following components:
1. resize.py We first resize images and annotations to 800 x 800 base
2. merge.py Group original DocHieNet annotations into DOSA compatible annotations, and merge into a single file
3. chunk.py split annotations into chunks
