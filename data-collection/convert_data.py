import pandas as pd
from tqdm import tqdm
from diff_utils import diff, syn
from py_formatter4dc import Formatter
def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Drop rows where 'message', 'origin_code', or 'patch' is empty
    df.dropna(inplace=True)

    # Optionally, reset the index after dropping rows
    df.reset_index(drop=True, inplace=True)
    return df

def phrase_1(df: pd.DataFrame,length_thre:int=6000) -> pd.DataFrame:
    phrase_1_df = df.copy()
    phrase_1_df.drop(columns=['message'], inplace=True)
    phrase_1_df.rename(columns={'origin_code': 'optimized_code'}, inplace=True)
    # phrase_1_df = phrase_1_df[phrase_1_df['optimized_code'].str.len() <= length_thre]
    tqdm.pandas()
    def process_row(row):
        try:
            old_code = syn(row['optimized_code'], row['patch'],reverse=True)
            return old_code
        except Exception as e:
            return ""
    phrase_1_df['old_code'] = phrase_1_df.progress_apply(process_row , axis=1)
    print(phrase_1_df.shape)
    phrase_1_df = clean_data(phrase_1_df)
    return phrase_1_df

def phrase_2(df: pd.DataFrame) -> pd.DataFrame:
    phrase_2_df = df.copy()
    formatter = Formatter()
    # 启用 tqdm 进度条
    tqdm.pandas()
    def process_row(row):
        try:
            cleaned_origin_code, cleaned_new_code = formatter.format(row['optimized_code'],row['old_code'] ,row['patch'])
        except Exception as e:
            print(e)
            return pd.Series(["",""])
        if not (cleaned_origin_code and cleaned_new_code):
            return pd.Series(["",""])
        patch = diff(cleaned_origin_code, cleaned_new_code)
        # instruction = prompt+"instruction:\n"+row['message']+"\n"+"old code:\n"+cleaned_origin_code
        return pd.Series([cleaned_origin_code,patch])
    phrase_2_df[['old_code', 'patch']] = phrase_2_df.progress_apply(process_row, axis=1)
    phrase_2_df = clean_data(phrase_2_df)
    return phrase_2_df

if __name__ == '__main__':
    
    df = pd.read_csv('/kaggle/input/github-commits/github_commit.csv', sep=',')
    df = clean_data(df)
    # df.drop(columns=['url'], inplace=True)
    # df = df.sample(frac=1).reset_index(drop=True)
    
    
    # Calculate the split index for 90% data
    split_index = int(len(df) * 0.9)
    output_dir="/kaggle/working/"
    # df_10 = df.iloc[split_index:]
    # df_10.to_json(output_dir+"test.json", orient='records')
    df_90 = df.iloc[:split_index]
    del df
    print(df_90.shape)
    phrase_1_df = phrase_1(df_90)
    phrase_1_df.to_json(output_dir+'train_phrase_1.json', orient='records')
    del phrase_1_df