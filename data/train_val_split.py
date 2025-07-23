"""
scripts.create_data_split.py
Dataset Training and Validation Splits
BoMeyering 2025
"""

import polars as pl
from glob import glob
from sklearn.model_selection import train_test_split

IMG_DIR = 'data/processed/images'

filenames = [img_name.split('.')[0] for img_name in glob('*.jpg', root_dir=IMG_DIR)]

if __name__ == '__main__':
    file_df = pl.DataFrame({'img_id': filenames})

    test_df, train_val_df = train_test_split(file_df, train_size=35)
    test_df = test_df.with_columns(
        split=pl.lit('test')
    )

    train_df, val_df = train_test_split(train_val_df, train_size=0.8)

    train_df = train_df.with_columns(
        split=pl.lit('train')
    )

    val_df = val_df.with_columns(
        split=pl.lit('val')
    )

    out_df = pl.concat([train_df, val_df, test_df])

    print(out_df)
    out_df.write_csv('data/data_split.csv')

